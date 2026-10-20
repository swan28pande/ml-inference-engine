// Size-class caching memory pool.
//
// Allocating device memory with cudaMalloc/cudaFree on every request costs
// tens to hundreds of microseconds and synchronises the device. MemoryPool
// rounds requests up to power-of-two size classes and keeps released blocks in
// per-class free lists so steady-state inference does zero driver
// allocations. Blocks can be pre-allocated at startup (Reserve) to remove
// warm-up misses entirely.
//
// The pool is allocator-agnostic: the same implementation manages pageable
// host memory, pinned host memory (cudaMallocHost) and device memory
// (cudaMalloc). CUDA-backed allocators are only available when built with
// MIE_WITH_CUDA.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mie/common/status.h"

namespace mie {

enum class MemoryKind { kHost, kPinnedHost, kDevice };
const char* MemoryKindName(MemoryKind kind);

// Raw allocation strategy wrapped by the pool.
class Allocator {
 public:
  virtual ~Allocator() = default;
  // Returns nullptr on failure.
  virtual void* Allocate(size_t bytes) = 0;
  virtual void Deallocate(void* ptr, size_t bytes) = 0;
  virtual MemoryKind kind() const = 0;
  virtual int device_id() const { return -1; }
};

// Aligned pageable host memory.
std::unique_ptr<Allocator> CreateHostAllocator(size_t alignment = 64);
// Page-locked host memory (cudaMallocHost). Unavailable without CUDA.
StatusOr<std::unique_ptr<Allocator>> CreatePinnedHostAllocator();
// Device memory on `device_id` (cudaMalloc). Unavailable without CUDA.
StatusOr<std::unique_ptr<Allocator>> CreateDeviceAllocator(int device_id);

struct MemoryPoolOptions {
  size_t min_block_bytes = 256;                    // smallest size class
  size_t max_block_bytes = size_t{256} << 20;      // larger requests bypass the cache
  size_t max_cached_bytes = size_t{1} << 30;       // cap on idle cached memory
};

struct MemoryPoolStats {
  uint64_t acquires = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t oversize_allocations = 0;
  uint64_t failed_allocations = 0;
  size_t bytes_in_use = 0;      // capacity of blocks currently handed out
  size_t bytes_cached = 0;      // capacity of idle blocks in free lists
  size_t peak_bytes_in_use = 0;

  double hit_rate() const { return acquires ? static_cast<double>(cache_hits) / acquires : 0.0; }
};

class MemoryPool;

// RAII handle to a pooled block. Returns the block to its pool on destruction.
// The pool must outlive all buffers it hands out.
class PooledBuffer {
 public:
  PooledBuffer() = default;
  ~PooledBuffer() { Release(); }
  PooledBuffer(PooledBuffer&& other) noexcept { *this = std::move(other); }
  PooledBuffer& operator=(PooledBuffer&& other) noexcept;
  PooledBuffer(const PooledBuffer&) = delete;
  PooledBuffer& operator=(const PooledBuffer&) = delete;

  void* data() const { return ptr_; }
  template <typename T> T* as() const { return static_cast<T*>(ptr_); }
  size_t size() const { return size_; }          // requested bytes
  size_t capacity() const { return capacity_; }  // size-class bytes
  explicit operator bool() const { return ptr_ != nullptr; }

  // Returns the block to the pool early.
  void Release();

 private:
  friend class MemoryPool;
  PooledBuffer(MemoryPool* pool, void* ptr, size_t size, size_t capacity)
      : pool_(pool), ptr_(ptr), size_(size), capacity_(capacity) {}

  MemoryPool* pool_ = nullptr;
  void* ptr_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

class MemoryPool {
 public:
  MemoryPool(std::unique_ptr<Allocator> allocator, MemoryPoolOptions options = {});
  ~MemoryPool();
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  // Returns a block of at least `bytes` bytes. Thread-safe.
  StatusOr<PooledBuffer> Acquire(size_t bytes);

  // Pre-allocates `count` blocks able to hold `bytes` each.
  Status Reserve(size_t bytes, size_t count);

  // Frees all idle cached blocks back to the underlying allocator.
  void Trim();

  MemoryPoolStats stats() const;
  MemoryKind kind() const { return allocator_->kind(); }
  const MemoryPoolOptions& options() const { return options_; }

  // Size class used for a request of `bytes` (0 = bypasses the cache).
  size_t SizeClassFor(size_t bytes) const;

 private:
  friend class PooledBuffer;

  struct Bucket {
    std::mutex mu;
    std::vector<void*> free_blocks;
  };

  void Return(void* ptr, size_t capacity);
  int BucketIndex(size_t size_class) const;
  void UpdatePeak(size_t in_use);

  std::unique_ptr<Allocator> allocator_;
  MemoryPoolOptions options_;
  std::vector<std::unique_ptr<Bucket>> buckets_;

  std::atomic<uint64_t> acquires_{0};
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> oversize_{0};
  std::atomic<uint64_t> failures_{0};
  std::atomic<size_t> bytes_in_use_{0};
  std::atomic<size_t> bytes_cached_{0};
  std::atomic<size_t> peak_in_use_{0};
};

// Process-wide device memory pools, one per GPU. Configure() must be called
// before the first Get() to take effect (typically from server startup).
class DeviceMemoryPools {
 public:
  static DeviceMemoryPools& Instance();
  void Configure(const MemoryPoolOptions& options);
  // Returns the pool for `device_id`, creating it on first use.
  StatusOr<MemoryPool*> Get(int device_id);
  // Returns the shared pinned host staging pool.
  StatusOr<MemoryPool*> PinnedHost();
  std::vector<std::pair<int, MemoryPoolStats>> AllStats() const;

 private:
  DeviceMemoryPools() = default;
  mutable std::mutex mu_;
  MemoryPoolOptions options_;
  std::vector<std::pair<int, std::unique_ptr<MemoryPool>>> pools_;
  std::unique_ptr<MemoryPool> pinned_;
};

}  // namespace mie
