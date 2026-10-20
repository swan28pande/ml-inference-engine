// Engine benchmark: measures latency percentiles and throughput of the
// scheduler + backend (no network) across max batch sizes.
//
// Closed loop (default): `concurrency` clients each keep one request in flight.
// Open loop (--rate R): requests arrive on a fixed schedule at R req/s and
// latency is measured from the scheduled send time, so queueing delay under
// overload is not hidden (no coordinated omission).
//
// Examples:
//   # simulated accelerator: 400us fixed cost per batch + 5us per sample
//   mie_engine_benchmark --backend reference --fixed_latency_us 400 --per_sample_latency_us 5
//   # real model on ONNX Runtime
//   mie_engine_benchmark --backend onnxruntime --model models/sample/mlp_v1.onnx --instances 2
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "benchmark_util.h"
#include "mie/common/logging.h"
#include "mie/engine/model_session.h"

namespace mie {
namespace {

struct RunResult {
  int max_batch = 0;
  double throughput = 0;
  double mean_batch = 0;
  uint64_t errors = 0;
  LatencyHistogram::Summary latency;
};

ModelConfig BuildConfig(const bench::Flags& flags, int max_batch) {
  ModelConfig c;
  c.name = "bench";
  c.version = "1";
  c.backend = flags.Str("backend", "reference");
  c.path = flags.Str("model", "");
  c.instance_count = static_cast<int>(flags.Int("instances", 1));
  c.use_gpu = flags.Has("gpu");
  c.warmup_iterations = 5;
  c.batching.enabled = max_batch > 1;
  c.batching.max_batch_size = max_batch;
  c.batching.max_queue_delay_us = flags.Int("delay_us", 500);
  c.batching.max_queue_size = 1 << 16;
  c.parameters["input_dim"] = flags.Str("input_dim", "16");
  c.parameters["output_dim"] = flags.Str("output_dim", "4");
  c.parameters["fixed_latency_us"] = flags.Str("fixed_latency_us", "0");
  c.parameters["per_sample_latency_us"] = flags.Str("per_sample_latency_us", "0");
  c.parameters["intra_op_threads"] = flags.Str("intra_op_threads", "1");
  if (flags.Has("execution_provider")) c.parameters["execution_provider"] = flags.Str("execution_provider", "cpu");
  return c;
}

TensorMap RandomInputs(const ModelMetadata& md, std::mt19937& rng) {
  TensorMap inputs = MakeZeroInputs(md.inputs, 1);
  std::uniform_real_distribution<float> dist(-1, 1);
  for (auto& [name, t] : inputs) {
    if (t.dtype() != DataType::kFloat32) continue;
    for (int64_t i = 0; i < t.num_elements(); ++i) t.data_as<float>()[i] = dist(rng);
  }
  return inputs;
}

RunResult RunClosedLoop(ModelSession& session, int concurrency, double duration_s) {
  LatencyHistogram latency;
  std::atomic<uint64_t> completed{0}, errors{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> clients;
  const auto start = Clock::now();
  for (int c = 0; c < concurrency; ++c) {
    clients.emplace_back([&, c] {
      std::mt19937 rng(static_cast<uint32_t>(c));
      TensorMap inputs = RandomInputs(session.metadata(), rng);
      while (!stop.load(std::memory_order_relaxed)) {
        const auto t0 = Clock::now();
        InferenceResult r = session.InferAsync(inputs).get();
        if (r.status.ok()) {
          latency.Record(MicrosSince(t0));
          completed.fetch_add(1, std::memory_order_relaxed);
        } else {
          errors.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::duration<double>(duration_s));
  stop.store(true);
  for (auto& t : clients) t.join();
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  RunResult r;
  r.throughput = static_cast<double>(completed.load()) / elapsed;
  r.errors = errors.load();
  r.latency = latency.Summarize();
  return r;
}

RunResult RunOpenLoop(ModelSession& session, double rate, double duration_s) {
  LatencyHistogram latency;
  std::atomic<uint64_t> completed{0}, errors{0}, outstanding{0};
  std::mt19937 rng(1);
  TensorMap inputs = RandomInputs(session.metadata(), rng);
  const auto interval = std::chrono::duration<double>(1.0 / rate);
  const auto start = Clock::now();
  const auto end = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(duration_s));
  uint64_t sent = 0;
  for (;;) {
    const auto scheduled = start + std::chrono::duration_cast<Clock::duration>(interval * static_cast<double>(sent));
    if (scheduled >= end) break;
    std::this_thread::sleep_until(scheduled);
    auto req = std::make_unique<InferenceRequest>();
    req->inputs = inputs;
    outstanding.fetch_add(1);
    req->callback = [&, scheduled](InferenceResult&& r) {
      if (r.status.ok()) {
        latency.Record(MicrosSince(scheduled));
        completed.fetch_add(1, std::memory_order_relaxed);
      } else {
        errors.fetch_add(1, std::memory_order_relaxed);
      }
      outstanding.fetch_sub(1);
    };
    session.Submit(std::move(req));
    ++sent;
  }
  while (outstanding.load() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  RunResult r;
  r.throughput = static_cast<double>(completed.load()) / elapsed;
  r.errors = errors.load();
  r.latency = latency.Summarize();
  return r;
}

}  // namespace
}  // namespace mie

int main(int argc, char** argv) {
  using namespace mie;
  bench::Flags flags(argc, argv);
  if (flags.Has("help")) {
    std::cout << "mie_engine_benchmark [--backend reference|onnxruntime|tensorrt] [--model PATH]\n"
                 "  [--batch_sizes 1,2,4,8,16,32,64] [--concurrency 64] [--duration_s 5] [--rate RPS]\n"
                 "  [--delay_us 500] [--instances 1] [--gpu]\n"
                 "  reference backend: [--input_dim 16] [--fixed_latency_us 0] [--per_sample_latency_us 0]\n";
    return 0;
  }
  SetLogLevel(LogLevel::WARNING);

  const auto batch_sizes = flags.IntList("batch_sizes", "1,2,4,8,16,32,64");
  const int concurrency = static_cast<int>(flags.Int("concurrency", 64));
  const double duration = flags.Double("duration_s", 5);
  const double rate = flags.Double("rate", 0);

  std::cout << "backend=" << flags.Str("backend", "reference") << " model=" << flags.Str("model", "<synthetic>")
            << " instances=" << flags.Int("instances", 1) << " delay_us=" << flags.Int("delay_us", 500)
            << (rate > 0 ? " mode=open-loop rate=" + std::to_string(static_cast<int>(rate))
                         : " mode=closed-loop concurrency=" + std::to_string(concurrency))
            << " duration=" << duration << "s\n\n";
  std::cout << "| max batch | mean batch | throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | errors |\n";
  std::cout << "|----------:|-----------:|-------------------:|---------:|---------:|---------:|-------:|\n";

  for (int max_batch : batch_sizes) {
    auto metrics = std::make_shared<ModelMetrics>("bench", "1");
    auto session = ModelSession::Create(BuildConfig(flags, max_batch), metrics);
    if (!session.ok()) {
      std::cerr << "failed to create session: " << session.status().ToString() << "\n";
      return 1;
    }
    RunResult r = rate > 0 ? RunOpenLoop(*session.value(), rate, duration)
                           : RunClosedLoop(*session.value(), concurrency, duration);
    session.value()->Shutdown(true);
    r.mean_batch = metrics->Snapshot().mean_batch_size;
    char mean_batch[16];
    std::snprintf(mean_batch, sizeof(mean_batch), "%.1f", r.mean_batch);
    std::cout << "| " << max_batch << " | " << mean_batch << " | " << bench::FormatRate(r.throughput) << " | "
              << bench::FormatMs(r.latency.p50) << " | " << bench::FormatMs(r.latency.p95) << " | "
              << bench::FormatMs(r.latency.p99) << " | " << r.errors << " |" << std::endl;
  }
  return 0;
}
