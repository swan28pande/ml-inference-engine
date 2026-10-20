#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "mie/engine/request_queue.h"

namespace mie {
namespace {

using std::chrono::microseconds;
using std::chrono::milliseconds;

InferenceRequestPtr MakeRequest(int64_t batch, int64_t dim = 4, DataType dtype = DataType::kFloat32) {
  auto r = std::make_unique<InferenceRequest>();
  r->inputs.emplace("input", Tensor(dtype, {batch, dim}));
  EXPECT_TRUE(PrepareRequest(r.get()).ok());
  r->enqueue_time = Clock::now();
  return r;
}

void Push(RequestQueue& q, InferenceRequestPtr r) { ASSERT_TRUE(q.Push(r).ok()); }

int64_t Samples(const std::vector<InferenceRequestPtr>& batch) {
  int64_t n = 0;
  for (const auto& r : batch) n += r->batch_size;
  return n;
}

TEST(PrepareRequestTest, ComputesBatchSizeAndSignature) {
  auto a = MakeRequest(3, 4);
  auto b = MakeRequest(1, 4);
  auto c = MakeRequest(1, 5);
  EXPECT_EQ(a->batch_size, 3);
  EXPECT_EQ(a->batch_signature, b->batch_signature);
  EXPECT_NE(a->batch_signature, c->batch_signature);
}

TEST(PrepareRequestTest, RejectsInconsistentBatchDims) {
  InferenceRequest r;
  r.inputs.emplace("a", Tensor(DataType::kFloat32, {2, 4}));
  r.inputs.emplace("b", Tensor(DataType::kFloat32, {3, 4}));
  EXPECT_EQ(PrepareRequest(&r).code(), StatusCode::kInvalidArgument);
  InferenceRequest empty;
  EXPECT_FALSE(PrepareRequest(&empty).ok());
}

TEST(RequestQueueTest, DispatchesImmediatelyWhenFull) {
  RequestQueue q(64);
  BatchPolicy policy{8, /*max_queue_delay_us=*/10'000'000, {}};
  for (int i = 0; i < 8; ++i) Push(q, MakeRequest(1));
  std::vector<InferenceRequestPtr> batch, expired;
  const auto start = Clock::now();
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_LT(MicrosSince(start), 100'000);  // did not wait for the 10s delay
  EXPECT_EQ(batch.size(), 8u);
  EXPECT_EQ(q.size(), 0u);
}

TEST(RequestQueueTest, DispatchesPartialBatchAfterDelay) {
  RequestQueue q(64);
  BatchPolicy policy{32, /*max_queue_delay_us=*/20'000, {}};
  Push(q, MakeRequest(1));
  Push(q, MakeRequest(2));
  std::vector<InferenceRequestPtr> batch, expired;
  const auto start = Clock::now();
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  const int64_t waited = MicrosSince(start);
  EXPECT_GE(waited, 15'000);
  EXPECT_LT(waited, 500'000);
  EXPECT_EQ(Samples(batch), 3);
}

TEST(RequestQueueTest, LateArrivalsJoinPendingBatch) {
  RequestQueue q(64);
  BatchPolicy policy{4, /*max_queue_delay_us=*/2'000'000, {}};
  Push(q, MakeRequest(1));
  std::thread producer([&] {
    for (int i = 0; i < 3; ++i) {
      std::this_thread::sleep_for(milliseconds(5));
      Push(q, MakeRequest(1));
    }
  });
  std::vector<InferenceRequestPtr> batch, expired;
  const auto start = Clock::now();
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  producer.join();
  EXPECT_EQ(batch.size(), 4u);
  EXPECT_LT(MicrosSince(start), 1'000'000);  // released by filling up, not by timeout
}

TEST(RequestQueueTest, NeverExceedsMaxBatchSize) {
  RequestQueue q(64);
  BatchPolicy policy{8, 0, {}};
  Push(q, MakeRequest(5));
  Push(q, MakeRequest(5));
  Push(q, MakeRequest(3));
  std::vector<InferenceRequestPtr> batch, expired;
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_EQ(Samples(batch), 5);  // 5 + 5 would overflow; FIFO order is preserved
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_EQ(Samples(batch), 8);
}

TEST(RequestQueueTest, GroupsOnlyCompatibleShapes) {
  RequestQueue q(64);
  BatchPolicy policy{16, 0, {}};
  Push(q, MakeRequest(1, 4));
  Push(q, MakeRequest(1, 8));
  Push(q, MakeRequest(1, 4));
  Push(q, MakeRequest(1, 4, DataType::kInt64));
  std::vector<InferenceRequestPtr> batch, expired;
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  ASSERT_EQ(batch.size(), 2u);
  for (const auto& r : batch) EXPECT_EQ(r->inputs.at("input").shape()[1], 4);
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0]->inputs.at("input").shape()[1], 8);
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0]->inputs.at("input").dtype(), DataType::kInt64);
}

TEST(RequestQueueTest, PreferredBatchSizesOnTimeout) {
  RequestQueue q(64);
  BatchPolicy policy{16, 1000, {4, 8, 16}};
  for (int i = 0; i < 6; ++i) Push(q, MakeRequest(1));
  std::vector<InferenceRequestPtr> batch, expired;
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_EQ(Samples(batch), 4);  // largest preferred size <= 6
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_EQ(Samples(batch), 2);  // remainder: below smallest preferred, take all
}

TEST(RequestQueueTest, RejectsWhenFull) {
  RequestQueue q(2);
  Push(q, MakeRequest(1));
  Push(q, MakeRequest(1));
  auto r = MakeRequest(1);
  Status s = q.Push(r);
  EXPECT_EQ(s.code(), StatusCode::kResourceExhausted);
  EXPECT_NE(r, nullptr);  // ownership stays with the caller on failure
}

TEST(RequestQueueTest, ExpiredRequestsAreSeparated) {
  RequestQueue q(8);
  BatchPolicy policy{8, 0, {}};
  auto stale = MakeRequest(1);
  stale->deadline = Clock::now() - milliseconds(1);
  Push(q, std::move(stale));
  Push(q, MakeRequest(1));
  std::vector<InferenceRequestPtr> batch, expired;
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_EQ(expired.size(), 1u);
  EXPECT_TRUE(batch.empty());
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_EQ(batch.size(), 1u);
}

TEST(RequestQueueTest, CloseWithDrainFlushesWithoutWaiting) {
  RequestQueue q(8);
  BatchPolicy policy{32, 10'000'000, {}};
  Push(q, MakeRequest(1));
  q.Close(/*drain=*/true);
  auto r = MakeRequest(1);
  EXPECT_EQ(q.Push(r).code(), StatusCode::kUnavailable);
  std::vector<InferenceRequestPtr> batch, expired;
  const auto start = Clock::now();
  ASSERT_TRUE(q.PopBatch(policy, &batch, &expired));
  EXPECT_LT(MicrosSince(start), 100'000);
  EXPECT_EQ(batch.size(), 1u);
  EXPECT_FALSE(q.PopBatch(policy, &batch, &expired));
}

TEST(RequestQueueTest, CloseWithoutDrainUnblocksWorkers) {
  RequestQueue q(8);
  BatchPolicy policy{32, 10'000'000, {}};
  std::thread worker([&] {
    std::vector<InferenceRequestPtr> batch, expired;
    EXPECT_FALSE(q.PopBatch(policy, &batch, &expired));
  });
  std::this_thread::sleep_for(milliseconds(10));
  q.Close(/*drain=*/false);
  worker.join();
}

}  // namespace
}  // namespace mie
