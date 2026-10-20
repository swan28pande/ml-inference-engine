#include "mie/memory/memory_pool.h"

#include <algorithm>
#include <cstdlib>

#include "mie/common/logging.h"
#include "mie/memory/cuda_utils.h"

#ifdef MIE_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mie {
namespace {

size_t NextPowerOfTwo(size_t v) {
  if (v <= 1) return 1;
  --v;
  for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1) v |= v >> shift;
  return v + 1;
}

int Log2(size_t power_of_two) {
  int n = 0;
  while (power_of_two > 1) {
    power_of_two >>= 1;
    ++n;
  }
  return n;
}

class HostAllocator final : public Allocator {
 public:
  explicit HostAllocator(size_t alignment) : alignment_(std::max<size_t>(alignment, sizeof(void*))) {}

  void* Allocate(size_t bytes) override {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, alignment_);
#else
    if (posix_memalign(&ptr, alignment_, bytes) != 0) ptr = nullptr;
#endif
    return ptr;
  }

  void Deallocate(void* ptr, size_t /*bytes*/) override {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
  }

  MemoryKind kind() const override { return MemoryKind::kHost; }

 private:
  size_t alignment_;
};

#ifdef MIE_WITH_CUDA
// Restores the calling thread's current device on scope exit.
class ScopedDevice {
 public:
  explicit ScopedDevice(int device) {
    cudaGetDevice(&previous_);
    if (previous_ != device) cudaSetDevice(device);
  }
  ~ScopedDevice() { cudaSetDevice(previous_); }

 private:
  int previous_ = 0;
};

class PinnedHostAllocator final : public Allocator {
 public:
  void* Allocate(size_t bytes) override {
    void* ptr = nullptr;
    if (cudaMallocHost(&ptr, bytes) != cudaSuccess) return nullptr;
    return ptr;
  }
  void Deallocate(void* ptr, size_t /*bytes*/) override { cudaFreeHost(ptr); }
  MemoryKind kind() const override { return MemoryKind::kPinnedHost; }
};

class DeviceAllocator final : public Allocator {
 public:
  explicit DeviceAllocator(int device_id) : device_id_(device_id) {}
  void* Allocate(size_t bytes) override {
    ScopedDevice guard(device_id_);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
      cudaGetLastError();  // clear the sticky error so later calls are not affected
      return nullptr;
    }
    return ptr;
  }
  void Deallocate(void* ptr, size_t /*bytes*/) override {
    ScopedDevice guard(device_id_);
    cudaFree(ptr);
  }
  MemoryKind kind() const override { return MemoryKind::kDevice; }
  int device_id() const override { return device_id_; }

 private:
  int device_id_;
};
#endif

}  // namespace

const char* MemoryKindName(MemoryKind kind) {
  switch (kind) {
    case MemoryKind::kHost: return "host";
    case MemoryKind::kPinnedHost: return "pinned_host";
    case MemoryKind::kDevice: return "device";
  }
  return "unknown";
}

std::unique_ptr<Allocator> CreateHostAllocator(size_t alignment) {
  return std::make_unique<HostAllocator>(alignment);
}

StatusOr<std::unique_ptr<Allocator>> CreatePinnedHostAllocator() {
#ifdef MIE_WITH_CUDA
  if (!CudaAvailable()) return Unavailable("no CUDA device available for pinned allocations");
  return std::unique_ptr<Allocator>(std::make_unique<PinnedHostAllocator>());
#else
  return Unavailable("built without CUDA support (MIE_WITH_CUDA=OFF)");
#endif
}

StatusOr<std::unique_ptr<Allocator>> CreateDeviceAllocator(int device_id) {
#ifdef MIE_WITH_CUDA
  if (device_id < 0 || device_id >= CudaDeviceCount()) {
    return InvalidArgument("CUDA device " + std::to_string(device_id) + " does not exist");
  }
  return std::unique_ptr<Allocator>(std::make_unique<DeviceAllocator>(device_id));
#else
  (void)device_id;
  return Unavailable("built without CUDA support (MIE_WITH_CUDA=OFF)");
#endif
}

// ------------------------------------------------------------------ PooledBuffer

PooledBuffer& PooledBuffer::operator=(PooledBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    pool_ = other.pool_;
    ptr_ = other.ptr_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.pool_ = nullptr;
    other.ptr_ = nullptr;
    other.size_ = other.capacity_ = 0;
  }
  return *this;
}

void PooledBuffer::Release() {
  if (pool_ && ptr_) pool_->Return(ptr_, capacity_);
  pool_ = nullptr;
  ptr_ = nullptr;
  size_ = capacity_ = 0;
}

// -------------------------------------------------------------------- MemoryPool

MemoryPool::MemoryPool(std::unique_ptr<Allocator> allocator, MemoryPoolOptions options)
    : allocator_(std::move(allocator)), options_(options) {
  options_.min_block_bytes = NextPowerOfTwo(std::max<size_t>(options_.min_block_bytes, 1));
  options_.max_block_bytes = NextPowerOfTwo(std::max(options_.max_block_bytes, options_.min_block_bytes));
  const int classes = Log2(options_.max_block_bytes) - Log2(options_.min_block_bytes) + 1;
  buckets_.reserve(static_cast<size_t>(classes));
  for (int i = 0; i < classes; ++i) buckets_.push_back(std::make_unique<Bucket>());
}

MemoryPool::~MemoryPool() {
  Trim();
  const size_t leaked = bytes_in_use_.load();
  if (leaked != 0) {
    MIE_LOG(WARNING) << "MemoryPool(" << MemoryKindName(kind()) << ") destroyed with " << leaked
                     << " bytes still in use";
  }
}

size_t MemoryPool::SizeClassFor(size_t bytes) const {
  if (bytes > options_.max_block_bytes) return 0;
  return std::max(NextPowerOfTwo(bytes), options_.min_block_bytes);
}

int MemoryPool::BucketIndex(size_t size_class) const {
  return Log2(size_class) - Log2(options_.min_block_bytes);
}

void MemoryPool::UpdatePeak(size_t in_use) {
  size_t peak = peak_in_use_.load(std::memory_order_relaxed);
  while (in_use > peak && !peak_in_use_.compare_exchange_weak(peak, in_use, std::memory_order_relaxed)) {
  }
}

StatusOr<PooledBuffer> MemoryPool::Acquire(size_t bytes) {
  acquires_.fetch_add(1, std::memory_order_relaxed);
  const size_t request = std::max<size_t>(bytes, 1);
  const size_t size_class = SizeClassFor(request);

  // Oversized requests bypass the cache: allocate exactly and free on release.
  if (size_class == 0) {
    void* ptr = allocator_->Allocate(request);
    if (!ptr) {
      Trim();
      ptr = allocator_->Allocate(request);
    }
    if (!ptr) {
      failures_.fetch_add(1, std::memory_order_relaxed);
      return ResourceExhausted("failed to allocate " + std::to_string(request) + " bytes of " +
                               MemoryKindName(kind()) + " memory");
    }
    oversize_.fetch_add(1, std::memory_order_relaxed);
    UpdatePeak(bytes_in_use_.fetch_add(request) + request);
    return PooledBuffer(this, ptr, bytes, request);
  }

  Bucket& bucket = *buckets_[static_cast<size_t>(BucketIndex(size_class))];
  void* ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(bucket.mu);
    if (!bucket.free_blocks.empty()) {
      ptr = bucket.free_blocks.back();
      bucket.free_blocks.pop_back();
    }
  }

  if (ptr) {
    hits_.fetch_add(1, std::memory_order_relaxed);
    bytes_cached_.fetch_sub(size_class, std::memory_order_relaxed);
  } else {
    misses_.fetch_add(1, std::memory_order_relaxed);
    ptr = allocator_->Allocate(size_class);
    if (!ptr) {
      // Out of memory: release idle cached blocks of other classes and retry.
      Trim();
      ptr = allocator_->Allocate(size_class);
    }
    if (!ptr) {
      failures_.fetch_add(1, std::memory_order_relaxed);
      return ResourceExhausted("failed to allocate " + std::to_string(size_class) + " bytes of " +
                               MemoryKindName(kind()) + " memory");
    }
  }
  UpdatePeak(bytes_in_use_.fetch_add(size_class) + size_class);
  return PooledBuffer(this, ptr, bytes, size_class);
}

void MemoryPool::Return(void* ptr, size_t capacity) {
  bytes_in_use_.fetch_sub(capacity, std::memory_order_relaxed);
  if (capacity > options_.max_block_bytes) {
    allocator_->Deallocate(ptr, capacity);
    return;
  }
  // Respect the idle-cache cap; free the block instead of caching it if full.
  const size_t cached = bytes_cached_.fetch_add(capacity) + capacity;
  if (cached > options_.max_cached_bytes) {
    bytes_cached_.fetch_sub(capacity);
    allocator_->Deallocate(ptr, capacity);
    return;
  }
  Bucket& bucket = *buckets_[static_cast<size_t>(BucketIndex(capacity))];
  std::lock_guard<std::mutex> lock(bucket.mu);
  bucket.free_blocks.push_back(ptr);
}

Status MemoryPool::Reserve(size_t bytes, size_t count) {
  const size_t size_class = SizeClassFor(std::max<size_t>(bytes, 1));
  if (size_class == 0) {
    return InvalidArgument("cannot reserve " + std::to_string(bytes) + " bytes: larger than max_block_bytes (" +
                           std::to_string(options_.max_block_bytes) + ")");
  }
  if (bytes_cached_.load() + size_class * count > options_.max_cached_bytes) {
    return InvalidArgument("reservation exceeds max_cached_bytes");
  }
  std::vector<void*> blocks;
  blocks.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    void* ptr = allocator_->Allocate(size_class);
    if (!ptr) {
      for (void* p : blocks) allocator_->Deallocate(p, size_class);
      return ResourceExhausted("pre-allocation of " + std::to_string(count) + " x " + std::to_string(size_class) +
                               " bytes failed");
    }
    blocks.push_back(ptr);
  }
  Bucket& bucket = *buckets_[static_cast<size_t>(BucketIndex(size_class))];
  std::lock_guard<std::mutex> lock(bucket.mu);
  bucket.free_blocks.insert(bucket.free_blocks.end(), blocks.begin(), blocks.end());
  bytes_cached_.fetch_add(size_class * count);
  return Status::Ok();
}

void MemoryPool::Trim() {
  for (size_t i = 0; i < buckets_.size(); ++i) {
    const size_t size_class = options_.min_block_bytes << i;
    std::vector<void*> blocks;
    {
      std::lock_guard<std::mutex> lock(buckets_[i]->mu);
      blocks.swap(buckets_[i]->free_blocks);
    }
    for (void* p : blocks) allocator_->Deallocate(p, size_class);
    bytes_cached_.fetch_sub(size_class * blocks.size());
  }
}

MemoryPoolStats MemoryPool::stats() const {
  MemoryPoolStats s;
  s.acquires = acquires_.load();
  s.cache_hits = hits_.load();
  s.cache_misses = misses_.load();
  s.oversize_allocations = oversize_.load();
  s.failed_allocations = failures_.load();
  s.bytes_in_use = bytes_in_use_.load();
  s.bytes_cached = bytes_cached_.load();
  s.peak_bytes_in_use = peak_in_use_.load();
  return s;
}

// ------------------------------------------------------------- DeviceMemoryPools

DeviceMemoryPools& DeviceMemoryPools::Instance() {
  static DeviceMemoryPools* instance = new DeviceMemoryPools();  // intentionally leaked (outlives CUDA users)
  return *instance;
}

void DeviceMemoryPools::Configure(const MemoryPoolOptions& options) {
  std::lock_guard<std::mutex> lock(mu_);
  options_ = options;
}

StatusOr<MemoryPool*> DeviceMemoryPools::Get(int device_id) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& [id, pool] : pools_) {
    if (id == device_id) return pool.get();
  }
  auto allocator = CreateDeviceAllocator(device_id);
  if (!allocator.ok()) return allocator.status();
  pools_.emplace_back(device_id, std::make_unique<MemoryPool>(std::move(allocator).value(), options_));
  return pools_.back().second.get();
}

StatusOr<MemoryPool*> DeviceMemoryPools::PinnedHost() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!pinned_) {
    auto allocator = CreatePinnedHostAllocator();
    if (!allocator.ok()) return allocator.status();
    pinned_ = std::make_unique<MemoryPool>(std::move(allocator).value(), options_);
  }
  return pinned_.get();
}

std::vector<std::pair<int, MemoryPoolStats>> DeviceMemoryPools::AllStats() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::pair<int, MemoryPoolStats>> out;
  for (const auto& [id, pool] : pools_) out.emplace_back(id, pool->stats());
  return out;
}

}  // namespace mie
