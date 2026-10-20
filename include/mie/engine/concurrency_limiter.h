// Caps the number of in-flight requests (admission control) and lets the
// server wait for them to drain during graceful shutdown.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace mie {

class ConcurrencyLimiter {
 public:
  // RAII permit; releases on destruction. Move-only.
  class Permit {
   public:
    Permit() = default;
    explicit Permit(ConcurrencyLimiter* owner) : owner_(owner) {}
    ~Permit() { Release(); }
    Permit(Permit&& o) noexcept : owner_(o.owner_) { o.owner_ = nullptr; }
    Permit& operator=(Permit&& o) noexcept {
      if (this != &o) {
        Release();
        owner_ = o.owner_;
        o.owner_ = nullptr;
      }
      return *this;
    }
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    explicit operator bool() const { return owner_ != nullptr; }
    void Release() {
      if (owner_) owner_->Release();
      owner_ = nullptr;
    }

   private:
    ConcurrencyLimiter* owner_ = nullptr;
  };

  // limit == 0 means unlimited.
  explicit ConcurrencyLimiter(size_t limit) : limit_(limit) {}

  // Non-blocking acquisition; returns an empty permit when at the limit.
  Permit TryAcquire() {
    std::lock_guard<std::mutex> lock(mu_);
    if (limit_ != 0 && in_flight_ >= limit_) return Permit();
    ++in_flight_;
    return Permit(this);
  }

  // Blocks until a permit is available or the timeout elapses.
  template <typename Rep, typename Period>
  Permit Acquire(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!cv_.wait_for(lock, timeout, [&] { return limit_ == 0 || in_flight_ < limit_; })) return Permit();
    ++in_flight_;
    return Permit(this);
  }

  // Waits until no permits are outstanding. Returns false on timeout.
  template <typename Rep, typename Period>
  bool WaitForIdle(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [&] { return in_flight_ == 0; });
  }

  size_t in_flight() const {
    std::lock_guard<std::mutex> lock(mu_);
    return in_flight_;
  }
  size_t limit() const { return limit_; }

 private:
  void Release() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      --in_flight_;
    }
    cv_.notify_all();
  }

  const size_t limit_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  size_t in_flight_ = 0;
};

}  // namespace mie
