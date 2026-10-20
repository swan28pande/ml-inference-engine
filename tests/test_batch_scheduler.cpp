#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "mie/engine/backends/reference_backend.h"
#include "mie/engine/model_session.h"

namespace mie {
namespace {

using std::chrono::milliseconds;

ModelConfig ReferenceConfig(int max_batch = 16, int64_t delay_us = 2000, int instances = 1) {
  ModelConfig c;
  c.name = "ref";
  c.version = "1";
  c.backend = "reference";
  c.instance_count = instances;
  c.batching.max_batch_size = max_batch;
  c.batching.max_queue_delay_us = delay_us;
  c.parameters["input_dim"] = "8";
  c.parameters["output_dim"] = "3";
  return c;
}

Tensor RandomInput(int64_t batch, int64_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::vector<float> v(static_cast<size_t>(batch * dim));
  for (auto& x : v) x = dist(rng);
  return Tensor::FromVector<float>({batch, dim}, v);
}

// Records every batch size the backend actually executes.
class RecordingBackend final : public ModelBackend {
 public:
  explicit RecordingBackend(std::vector<int>* sizes, std::mutex* mu, int sleep_us = 0)
      : sizes_(sizes), mu_(mu), sleep_us_(sleep_us) {}
  Status Load(const ModelConfig& config) override {
    md_.max_batch_size = config.batching.max_batch_size;
    md_.inputs = {{"x", DataType::kFloat32, {-1, 1}}};
    md_.outputs = {{"y", DataType::kFloat32, {-1, 1}}};
    return Status::Ok();
  }
  Status Infer(const TensorMap& inputs, TensorMap* outputs) override {
    const Tensor& x = inputs.at("x");
    {
      std::lock_guard<std::mutex> lock(*mu_);
      sizes_->push_back(static_cast<int>(x.shape()[0]));
    }
    if (sleep_us_) std::this_thread::sleep_for(std::chrono::microseconds(sleep_us_));
    Tensor y = x;  // identity
    for (int64_t i = 0; i < y.num_elements(); ++i) y.data_as<float>()[i] *= 2.0f;
    (*outputs)["y"] = std::move(y);
    return Status::Ok();
  }
  const ModelMetadata& metadata() const override { return md_; }

 private:
  ModelMetadata md_;
  std::vector<int>* sizes_;
  std::mutex* mu_;
  int sleep_us_;
};

TEST(BatchSchedulerTest, BatchedResultsMatchUnbatchedComputation) {
  ModelConfig cfg = ReferenceConfig(/*max_batch=*/32, /*delay_us=*/5000);
  auto session = ModelSession::Create(cfg, nullptr);
  ASSERT_TRUE(session.ok()) << session.status().ToString();

  ReferenceBackend oracle;
  ASSERT_TRUE(oracle.Load(cfg).ok());

  constexpr int kRequests = 64;
  std::vector<Tensor> inputs;
  std::vector<std::future<InferenceResult>> futures;
  for (int i = 0; i < kRequests; ++i) {
    const int64_t b = 1 + (i % 3);
    inputs.push_back(RandomInput(b, 8, static_cast<uint32_t>(i)));
    TensorMap m;
    m.emplace("input", inputs.back());
    futures.push_back(session.value()->InferAsync(std::move(m)));
  }

  int max_seen_batch = 0;
  for (int i = 0; i < kRequests; ++i) {
    InferenceResult r = futures[static_cast<size_t>(i)].get();
    ASSERT_TRUE(r.status.ok()) << r.status.ToString();
    const Tensor& out = r.outputs.at("output");
    const int64_t b = inputs[static_cast<size_t>(i)].shape()[0];
    ASSERT_EQ(out.shape(), (Shape{b, 3}));
    for (int64_t s = 0; s < b; ++s) {
      const auto expected = oracle.Forward(inputs[static_cast<size_t>(i)].data_as<float>() + s * 8);
      for (int o = 0; o < 3; ++o) EXPECT_FLOAT_EQ(out.data_as<float>()[s * 3 + o], expected[static_cast<size_t>(o)]);
    }
    max_seen_batch = std::max(max_seen_batch, r.timing.batch_size);
    EXPECT_LE(r.timing.batch_size, 32);
  }
  EXPECT_GT(max_seen_batch, 3) << "requests were never batched together";
}

TEST(BatchSchedulerTest, FormsLargeBatchesUnderConcurrentLoad) {
  std::vector<int> sizes;
  std::mutex mu;
  RecordingBackend backend(&sizes, &mu, /*sleep_us=*/2000);
  ModelConfig cfg;
  cfg.batching.max_batch_size = 16;
  cfg.batching.max_queue_delay_us = 1000;
  ASSERT_TRUE(backend.Load(cfg).ok());
  auto metrics = std::make_shared<ModelMetrics>("m", "1");
  BatchScheduler scheduler(cfg.batching, 16, {&backend}, metrics);
  scheduler.Start();

  constexpr int kThreads = 16;
  constexpr int kPerThread = 50;
  std::atomic<int> ok{0};
  std::vector<std::thread> clients;
  for (int t = 0; t < kThreads; ++t) {
    clients.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i) {
        std::promise<InferenceResult> p;
        auto f = p.get_future();
        auto r = std::make_unique<InferenceRequest>();
        const float value = static_cast<float>(t * 1000 + i);
        r->inputs.emplace("x", Tensor::FromVector<float>({1, 1}, {value}));
        r->callback = [&p](InferenceResult&& res) { p.set_value(std::move(res)); };
        ASSERT_TRUE(scheduler.Enqueue(r).ok());
        InferenceResult res = f.get();
        ASSERT_TRUE(res.status.ok());
        // Each request must get back exactly its own slice.
        ASSERT_FLOAT_EQ(res.outputs.at("y").data_as<float>()[0], value * 2.0f);
        ok.fetch_add(1);
      }
    });
  }
  for (auto& c : clients) c.join();
  scheduler.Stop(true);

  EXPECT_EQ(ok.load(), kThreads * kPerThread);
  int total = 0, max_size = 0;
  for (int s : sizes) {
    total += s;
    max_size = std::max(max_size, s);
    EXPECT_LE(s, 16);
  }
  EXPECT_EQ(total, kThreads * kPerThread);
  const double mean = static_cast<double>(total) / static_cast<double>(sizes.size());
  EXPECT_GT(mean, 4.0) << "dynamic batching should amortise the 2ms backend cost";
  EXPECT_EQ(max_size, 16);

  const auto snap = metrics->Snapshot();
  EXPECT_EQ(snap.requests_ok, static_cast<uint64_t>(kThreads * kPerThread));
  EXPECT_GT(snap.mean_batch_size, 4.0);
  EXPECT_GT(snap.request_latency.p99, 0.0);
}

TEST(BatchSchedulerTest, BatchingDisabledRunsOneRequestAtATime) {
  std::vector<int> sizes;
  std::mutex mu;
  RecordingBackend backend(&sizes, &mu);
  ModelConfig cfg;
  cfg.batching.enabled = false;
  ASSERT_TRUE(backend.Load(cfg).ok());
  BatchScheduler scheduler(cfg.batching, 32, {&backend}, nullptr);
  EXPECT_EQ(scheduler.policy().max_batch_size, 1);
  scheduler.Start();
  std::vector<std::future<InferenceResult>> fs;
  std::vector<std::shared_ptr<std::promise<InferenceResult>>> ps;
  for (int i = 0; i < 10; ++i) {
    auto p = std::make_shared<std::promise<InferenceResult>>();
    fs.push_back(p->get_future());
    auto r = std::make_unique<InferenceRequest>();
    r->inputs.emplace("x", Tensor::FromVector<float>({1, 1}, {1.0f}));
    r->callback = [p](InferenceResult&& res) { p->set_value(std::move(res)); };
    ASSERT_TRUE(scheduler.Enqueue(r).ok());
  }
  for (auto& f : fs) EXPECT_TRUE(f.get().status.ok());
  scheduler.Stop(true);
  for (int s : sizes) EXPECT_EQ(s, 1);
}

TEST(BatchSchedulerTest, RejectsOversizedRequests) {
  auto session = ModelSession::Create(ReferenceConfig(/*max_batch=*/4), nullptr);
  ASSERT_TRUE(session.ok());
  TensorMap m;
  m.emplace("input", RandomInput(8, 8, 1));
  auto r = session.value()->InferAsync(std::move(m)).get();
  EXPECT_EQ(r.status.code(), StatusCode::kInvalidArgument);
}

TEST(BatchSchedulerTest, OutputFilter) {
  auto session = ModelSession::Create(ReferenceConfig(), nullptr);
  ASSERT_TRUE(session.ok());
  TensorMap m;
  m.emplace("input", RandomInput(1, 8, 1));
  auto bad = session.value()->InferAsync(m, {"does_not_exist"}).get();
  EXPECT_EQ(bad.status.code(), StatusCode::kInvalidArgument);
  auto good = session.value()->InferAsync(m, {"output"}).get();
  EXPECT_TRUE(good.status.ok());
  EXPECT_EQ(good.outputs.size(), 1u);
}

TEST(BatchSchedulerTest, ExpiredDeadlinesFailWithoutExecuting) {
  ModelConfig cfg = ReferenceConfig(/*max_batch=*/8, /*delay_us=*/0);
  cfg.parameters["fixed_latency_us"] = "50000";  // keep the single worker busy
  auto session = ModelSession::Create(cfg, nullptr);
  ASSERT_TRUE(session.ok());
  TensorMap m;
  m.emplace("input", RandomInput(8, 8, 1));
  auto busy = session.value()->InferAsync(m);  // occupies the worker for 50ms
  std::this_thread::sleep_for(milliseconds(5));
  TensorMap m2;
  m2.emplace("input", RandomInput(1, 8, 2));
  auto doomed = session.value()->InferAsync(std::move(m2), {}, Clock::now() + milliseconds(10));
  EXPECT_EQ(doomed.get().status.code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(busy.get().status.ok());
}

TEST(BatchSchedulerTest, QueueFullIsRejected) {
  ModelConfig cfg = ReferenceConfig(/*max_batch=*/1, /*delay_us=*/0);
  cfg.batching.max_queue_size = 2;
  cfg.parameters["fixed_latency_us"] = "100000";
  auto session = ModelSession::Create(cfg, nullptr);
  ASSERT_TRUE(session.ok());
  std::vector<std::future<InferenceResult>> fs;
  for (int i = 0; i < 6; ++i) {
    TensorMap m;
    m.emplace("input", RandomInput(1, 8, static_cast<uint32_t>(i)));
    fs.push_back(session.value()->InferAsync(std::move(m)));
  }
  int rejected = 0;
  for (auto& f : fs) rejected += f.get().status.code() == StatusCode::kResourceExhausted;
  EXPECT_GE(rejected, 3);
  EXPECT_GE(session.value()->metrics()->Snapshot().requests_rejected, 3u);
}

TEST(BatchSchedulerTest, ShutdownWithDrainCompletesAllRequests) {
  ModelConfig cfg = ReferenceConfig(/*max_batch=*/4, /*delay_us=*/1000, /*instances=*/2);
  cfg.parameters["fixed_latency_us"] = "2000";
  auto session = ModelSession::Create(cfg, nullptr);
  ASSERT_TRUE(session.ok());
  std::vector<std::future<InferenceResult>> fs;
  for (int i = 0; i < 40; ++i) {
    TensorMap m;
    m.emplace("input", RandomInput(1, 8, static_cast<uint32_t>(i)));
    fs.push_back(session.value()->InferAsync(std::move(m)));
  }
  session.value()->Shutdown(/*drain=*/true);
  for (auto& f : fs) {
    ASSERT_EQ(f.wait_for(milliseconds(0)), std::future_status::ready) << "callback not invoked before Shutdown returned";
    EXPECT_TRUE(f.get().status.ok());
  }
  TensorMap m;
  m.emplace("input", RandomInput(1, 8, 99));
  EXPECT_EQ(session.value()->InferAsync(std::move(m)).get().status.code(), StatusCode::kUnavailable);
}

TEST(BatchSchedulerTest, ShutdownWithoutDrainFailsQueuedRequests) {
  ModelConfig cfg = ReferenceConfig(/*max_batch=*/1, /*delay_us=*/0);
  cfg.parameters["fixed_latency_us"] = "20000";
  auto session = ModelSession::Create(cfg, nullptr);
  ASSERT_TRUE(session.ok());
  std::vector<std::future<InferenceResult>> fs;
  for (int i = 0; i < 10; ++i) {
    TensorMap m;
    m.emplace("input", RandomInput(1, 8, static_cast<uint32_t>(i)));
    fs.push_back(session.value()->InferAsync(std::move(m)));
  }
  session.value()->Shutdown(/*drain=*/false);
  int unavailable = 0;
  for (auto& f : fs) {
    ASSERT_EQ(f.wait_for(milliseconds(0)), std::future_status::ready);
    unavailable += f.get().status.code() == StatusCode::kUnavailable;
  }
  EXPECT_GE(unavailable, 8);
}

}  // namespace
}  // namespace mie
