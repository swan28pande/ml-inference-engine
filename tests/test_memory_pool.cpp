#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "mie/memory/cuda_utils.h"
#include "mie/memory/memory_pool.h"

namespace mie {
namespace {

// Counts calls into the underlying allocator to verify caching behaviour.
class CountingAllocator final : public Allocator {
 public:
  explicit CountingAllocator(std::atomic<int>* allocs, std::atomic<int>* frees, size_t fail_above = SIZE_MAX)
      : inner_(CreateHostAllocator()), allocs_(allocs), frees_(frees), fail_above_(fail_above) {}
  void* Allocate(size_t bytes) override {
    if (bytes > fail_above_) return nullptr;
    allocs_->fetch_add(1);
    return inner_->Allocate(bytes);
  }
  void Deallocate(void* p, size_t bytes) override {
    frees_->fetch_add(1);
    inner_->Deallocate(p, bytes);
  }
  MemoryKind kind() const override { return MemoryKind::kHost; }

 private:
  std::unique_ptr<Allocator> inner_;
  std::atomic<int>* allocs_;
  std::atomic<int>* frees_;
  size_t fail_above_;
};

class MemoryPoolTest : public ::testing::Test {
 protected:
  std::unique_ptr<MemoryPool> MakePool(MemoryPoolOptions opts = {}, size_t fail_above = SIZE_MAX) {
    return std::make_unique<MemoryPool>(std::make_unique<CountingAllocator>(&allocs_, &frees_, fail_above), opts);
  }
  std::atomic<int> allocs_{0};
  std::atomic<int> frees_{0};
};

TEST_F(MemoryPoolTest, RoundsUpToPowerOfTwoSizeClasses) {
  MemoryPoolOptions opts;
  opts.min_block_bytes = 256;
  auto pool = MakePool(opts);
  EXPECT_EQ(pool->SizeClassFor(1), 256u);
  EXPECT_EQ(pool->SizeClassFor(256), 256u);
  EXPECT_EQ(pool->SizeClassFor(257), 512u);
  EXPECT_EQ(pool->SizeClassFor(5000), 8192u);
  EXPECT_EQ(pool->SizeClassFor(opts.max_block_bytes + 1), 0u);  // bypasses cache

  auto buf = pool->Acquire(1000);
  ASSERT_TRUE(buf.ok());
  EXPECT_EQ(buf->size(), 1000u);
  EXPECT_EQ(buf->capacity(), 1024u);
}

TEST_F(MemoryPoolTest, ReusesReleasedBlocks) {
  auto pool = MakePool();
  void* first = nullptr;
  {
    auto buf = pool->Acquire(4096);
    ASSERT_TRUE(buf.ok());
    first = buf->data();
    std::memset(buf->data(), 0xAB, 4096);
  }
  EXPECT_EQ(allocs_.load(), 1);
  EXPECT_EQ(pool->stats().bytes_cached, 4096u);
  {
    auto buf = pool->Acquire(3000);  // same 4 KiB size class
    ASSERT_TRUE(buf.ok());
    EXPECT_EQ(buf->data(), first);
  }
  EXPECT_EQ(allocs_.load(), 1);  // no second driver allocation
  const auto s = pool->stats();
  EXPECT_EQ(s.acquires, 2u);
  EXPECT_EQ(s.cache_hits, 1u);
  EXPECT_EQ(s.cache_misses, 1u);
  EXPECT_DOUBLE_EQ(s.hit_rate(), 0.5);
}

TEST_F(MemoryPoolTest, ReservePreallocatesBlocks) {
  auto pool = MakePool();
  ASSERT_TRUE(pool->Reserve(65536, 8).ok());
  EXPECT_EQ(allocs_.load(), 8);
  std::vector<PooledBuffer> held;
  for (int i = 0; i < 8; ++i) {
    auto b = pool->Acquire(65536);
    ASSERT_TRUE(b.ok());
    held.push_back(std::move(b).value());
  }
  EXPECT_EQ(allocs_.load(), 8);  // all served from the reservation
  EXPECT_EQ(pool->stats().cache_hits, 8u);
  EXPECT_EQ(pool->stats().bytes_in_use, 8u * 65536u);
}

TEST_F(MemoryPoolTest, TracksInUseAndPeak) {
  auto pool = MakePool();
  {
    auto a = pool->Acquire(1024);
    auto b = pool->Acquire(2048);
    EXPECT_EQ(pool->stats().bytes_in_use, 3072u);
  }
  EXPECT_EQ(pool->stats().bytes_in_use, 0u);
  EXPECT_EQ(pool->stats().peak_bytes_in_use, 3072u);
}

TEST_F(MemoryPoolTest, OversizeRequestsBypassCache) {
  MemoryPoolOptions opts;
  opts.max_block_bytes = 4096;
  auto pool = MakePool(opts);
  {
    auto b = pool->Acquire(10000);
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(b->capacity(), 10000u);
  }
  EXPECT_EQ(frees_.load(), 1);  // freed immediately, not cached
  EXPECT_EQ(pool->stats().oversize_allocations, 1u);
  EXPECT_EQ(pool->stats().bytes_cached, 0u);
}

TEST_F(MemoryPoolTest, RespectsMaxCachedBytes) {
  MemoryPoolOptions opts;
  opts.max_cached_bytes = 8192;
  auto pool = MakePool(opts);
  {
    auto a = pool->Acquire(4096);
    auto b = pool->Acquire(4096);
    auto c = pool->Acquire(4096);
  }
  EXPECT_LE(pool->stats().bytes_cached, 8192u);
  EXPECT_EQ(frees_.load(), 1);  // third block exceeded the cap
}

TEST_F(MemoryPoolTest, TrimReleasesIdleBlocks) {
  auto pool = MakePool();
  { auto a = pool->Acquire(1024); }
  EXPECT_GT(pool->stats().bytes_cached, 0u);
  pool->Trim();
  EXPECT_EQ(pool->stats().bytes_cached, 0u);
  EXPECT_EQ(frees_.load(), 1);
}

TEST_F(MemoryPoolTest, AllocationFailureIsReported) {
  auto pool = MakePool({}, /*fail_above=*/1024);
  auto ok = pool->Acquire(512);
  EXPECT_TRUE(ok.ok());
  auto fail = pool->Acquire(1 << 20);
  EXPECT_FALSE(fail.ok());
  EXPECT_EQ(fail.status().code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(pool->stats().failed_allocations, 1u);
}

TEST_F(MemoryPoolTest, MoveTransfersOwnership) {
  auto pool = MakePool();
  auto a = pool->Acquire(1024);
  ASSERT_TRUE(a.ok());
  PooledBuffer b = std::move(a).value();
  EXPECT_TRUE(static_cast<bool>(b));
  PooledBuffer c;
  c = std::move(b);
  EXPECT_FALSE(static_cast<bool>(b));  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(static_cast<bool>(c));
  c.Release();
  EXPECT_EQ(pool->stats().bytes_in_use, 0u);
}

TEST_F(MemoryPoolTest, ConcurrentAcquireRelease) {
  auto pool = MakePool();
  constexpr int kThreads = 8;
  constexpr int kIters = 2000;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        const size_t size = 256u << ((t + i) % 6);
        auto b = pool->Acquire(size);
        ASSERT_TRUE(b.ok());
        std::memset(b->data(), t, size);
      }
    });
  }
  for (auto& th : threads) th.join();
  const auto s = pool->stats();
  EXPECT_EQ(s.acquires, static_cast<uint64_t>(kThreads * kIters));
  EXPECT_EQ(s.bytes_in_use, 0u);
  // Steady state: almost everything is a cache hit.
  EXPECT_GT(s.hit_rate(), 0.95);
  EXPECT_LE(allocs_.load(), kThreads * 6);
}

TEST(DeviceMemoryPoolTest, UnavailableWithoutGpu) {
  if (CudaAvailable()) GTEST_SKIP() << "GPU present";
  auto pool = DeviceMemoryPools::Instance().Get(0);
  EXPECT_FALSE(pool.ok());
}

#ifdef MIE_WITH_CUDA
TEST(DeviceMemoryPoolTest, ReusesDeviceBlocks) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
  auto pool = DeviceMemoryPools::Instance().Get(0);
  ASSERT_TRUE(pool.ok());
  void* first = nullptr;
  {
    auto b = pool.value()->Acquire(1 << 20);
    ASSERT_TRUE(b.ok());
    first = b->data();
  }
  auto b = pool.value()->Acquire(1 << 20);
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(b->data(), first);
}
#endif

}  // namespace
}  // namespace mie
