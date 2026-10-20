// mie_loadgen: closed-loop gRPC load generator for a running mie_server.
//
//   mie_loadgen --address localhost:50051 --model mlp --concurrency 64 --duration_s 10
//
// Spreads clients over several independent HTTP/2 connections (--channels),
// reads the input signature from GetModelMetadata and reports client-observed
// latency percentiles (network + queueing + compute) and throughput.
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "benchmark_util.h"
#include "inference.grpc.pb.h"

int main(int argc, char** argv) {
  using namespace mie;
  bench::Flags flags(argc, argv);
  if (flags.Has("help")) {
    std::cout << "mie_loadgen [--address localhost:50051] [--model mlp] [--version V] [--concurrency 64]\n"
                 "            [--channels 8] [--duration_s 10] [--warmup_s 2] [--batch 1]\n";
    return 0;
  }
  const std::string address = flags.Str("address", "localhost:50051");
  const std::string model = flags.Str("model", "mlp");
  const std::string version = flags.Str("version", "");
  const int concurrency = static_cast<int>(flags.Int("concurrency", 64));
  const int num_channels = static_cast<int>(flags.Int("channels", 8));
  const double duration = flags.Double("duration_s", 10);
  const double warmup = flags.Double("warmup_s", 2);
  const int64_t batch = flags.Int("batch", 1);

  std::vector<std::shared_ptr<grpc::Channel>> channels;
  for (int i = 0; i < num_channels; ++i) {
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);  // force a distinct connection per channel
    args.SetInt("mie.channel_id", i);
    channels.push_back(grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args));
  }

  // Discover the input signature.
  v1::ModelMetadataResponse md;
  {
    auto stub = v1::InferenceService::NewStub(channels[0]);
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    v1::ModelMetadataRequest req;
    req.set_model_name(model);
    req.set_model_version(version);
    const auto s = stub->GetModelMetadata(&ctx, req, &md);
    if (!s.ok()) {
      std::cerr << "GetModelMetadata failed: " << s.error_message() << "\n";
      return 1;
    }
  }

  v1::PredictRequest prototype;
  prototype.set_model_name(model);
  prototype.set_model_version(version);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1, 1);
  for (const auto& spec : md.inputs()) {
    if (spec.dtype() != v1::DT_FLOAT32) {
      std::cerr << "only float32 inputs are supported by the load generator\n";
      return 1;
    }
    auto* t = prototype.add_inputs();
    t->set_name(spec.name());
    t->set_dtype(spec.dtype());
    int64_t n = 1;
    for (int i = 0; i < spec.shape_size(); ++i) {
      const int64_t d = spec.shape(i) < 0 ? (i == 0 ? batch : 1) : spec.shape(i);
      t->add_shape(d);
      n *= d;
    }
    std::vector<float> data(static_cast<size_t>(n));
    for (auto& v : data) v = dist(rng);
    t->set_raw_data(data.data(), data.size() * sizeof(float));
  }

  LatencyHistogram latency, server_total, server_queue;
  std::atomic<uint64_t> ok{0}, failed{0}, batch_sum{0};
  std::atomic<bool> measuring{false}, stop{false};
  std::vector<std::thread> clients;
  for (int c = 0; c < concurrency; ++c) {
    clients.emplace_back([&, c] {
      auto stub = v1::InferenceService::NewStub(channels[static_cast<size_t>(c % num_channels)]);
      v1::PredictRequest req = prototype;
      req.set_routing_key("client-" + std::to_string(c));
      v1::PredictResponse resp;
      while (!stop.load(std::memory_order_relaxed)) {
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        resp.Clear();
        const auto t0 = Clock::now();
        const grpc::Status s = stub->Predict(&ctx, req, &resp);
        const int64_t us = MicrosSince(t0);
        if (!measuring.load(std::memory_order_relaxed)) continue;
        if (s.ok()) {
          latency.Record(us);
          server_total.Record(resp.timing().total_us());
          server_queue.Record(resp.timing().queue_us());
          batch_sum.fetch_add(static_cast<uint64_t>(resp.timing().batch_size()), std::memory_order_relaxed);
          ok.fetch_add(1, std::memory_order_relaxed);
        } else {
          failed.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::duration<double>(warmup));
  measuring.store(true);
  const auto start = Clock::now();
  std::this_thread::sleep_for(std::chrono::duration<double>(duration));
  measuring.store(false);
  const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  stop.store(true);
  for (auto& t : clients) t.join();

  const auto l = latency.Summarize();
  const auto st = server_total.Summarize();
  const auto sq = server_queue.Summarize();
  const double rps = static_cast<double>(ok.load()) / elapsed;
  std::cout << "target=" << address << " model=" << model << " concurrency=" << concurrency
            << " channels=" << num_channels << " batch=" << batch << " duration=" << duration << "s\n\n";
  std::cout << "| metric | value |\n|---|---:|\n";
  std::cout << "| throughput (req/s) | " << bench::FormatRate(rps) << " |\n";
  std::cout << "| throughput (samples/s) | " << bench::FormatRate(rps * static_cast<double>(batch)) << " |\n";
  std::cout << "| client p50 / p95 / p99 (ms) | " << bench::FormatMs(l.p50) << " / " << bench::FormatMs(l.p95)
            << " / " << bench::FormatMs(l.p99) << " |\n";
  std::cout << "| server p50 / p99 (ms) | " << bench::FormatMs(st.p50) << " / " << bench::FormatMs(st.p99) << " |\n";
  std::cout << "| server queue p50 / p99 (ms) | " << bench::FormatMs(sq.p50) << " / " << bench::FormatMs(sq.p99)
            << " |\n";
  char mean_batch[16];
  std::snprintf(mean_batch, sizeof(mean_batch), "%.1f",
                ok.load() ? static_cast<double>(batch_sum.load()) / static_cast<double>(ok.load()) : 0.0);
  std::cout << "| mean executed batch | " << mean_batch << " |\n";
  std::cout << "| errors | " << failed.load() << " |\n";
  return failed.load() > 0 && ok.load() == 0 ? 1 : 0;
}
