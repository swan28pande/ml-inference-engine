#include "mie/engine/request_queue.h"

#include <algorithm>

namespace mie {

RequestQueue::RequestQueue(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

Status RequestQueue::Push(InferenceRequestPtr& request) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) return Unavailable("model is shutting down");
    if (queue_.size() >= capacity_) {
      return ResourceExhausted("request queue full (" + std::to_string(capacity_) + " pending)");
    }
    queue_.push_back(std::move(request));
  }
  cv_.notify_one();
  return Status::Ok();
}

void RequestQueue::RemoveExpired(TimePoint now, std::vector<InferenceRequestPtr>* expired) {
  for (auto it = queue_.begin(); it != queue_.end();) {
    if ((*it)->deadline && *(*it)->deadline <= now) {
      expired->push_back(std::move(*it));
      it = queue_.erase(it);
    } else {
      ++it;
    }
  }
}

// Number of samples that could join the head request's batch, stopping at the
// first compatible request that would overflow max_batch_size.
int64_t RequestQueue::CompatibleSamples(int max_batch_size) const {
  const std::string& sig = queue_.front()->batch_signature;
  int64_t total = 0;
  for (const auto& r : queue_) {
    if (r->batch_signature != sig) continue;
    if (total + r->batch_size > max_batch_size) return max_batch_size;  // cannot grow further
    total += r->batch_size;
  }
  return total;
}

void RequestQueue::TakeCompatible(const BatchPolicy& policy, bool timed_out,
                                  std::vector<InferenceRequestPtr>* batch) {
  int64_t target = policy.max_batch_size;
  if (timed_out && !policy.preferred_batch_sizes.empty()) {
    const int64_t available = CompatibleSamples(policy.max_batch_size);
    for (auto it = policy.preferred_batch_sizes.rbegin(); it != policy.preferred_batch_sizes.rend(); ++it) {
      if (*it <= available) {
        target = *it;
        break;
      }
    }
  }

  // The head is always taken so progress is guaranteed.
  const std::string sig = queue_.front()->batch_signature;
  int64_t taken = queue_.front()->batch_size;
  batch->push_back(std::move(queue_.front()));
  queue_.pop_front();

  for (auto it = queue_.begin(); it != queue_.end() && taken < target;) {
    if ((*it)->batch_signature != sig) {
      ++it;
      continue;
    }
    if (taken + (*it)->batch_size > target) break;  // preserve FIFO order within a signature
    taken += (*it)->batch_size;
    batch->push_back(std::move(*it));
    it = queue_.erase(it);
  }
}

bool RequestQueue::PopBatch(const BatchPolicy& policy, std::vector<InferenceRequestPtr>* batch,
                            std::vector<InferenceRequestPtr>* expired) {
  batch->clear();
  expired->clear();
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    cv_.wait(lock, [&] { return !queue_.empty() || closed_; });
    if (closed_ && !drain_) return false;

    const TimePoint now = Clock::now();
    RemoveExpired(now, expired);
    if (!expired->empty()) return true;  // let the caller fail them promptly
    if (queue_.empty()) {
      if (closed_) return false;
      continue;
    }

    const bool full = CompatibleSamples(policy.max_batch_size) >= policy.max_batch_size;
    const TimePoint deadline = queue_.front()->enqueue_time + std::chrono::microseconds(policy.max_queue_delay_us);
    const bool timed_out = now >= deadline;

    // While draining we never wait for a batch to fill.
    if (full || timed_out || closed_) {
      TakeCompatible(policy, timed_out && !full, batch);
      if (!queue_.empty()) cv_.notify_one();  // hand the remainder to another worker
      return true;
    }

    // Wake up at the head's deadline or on a new arrival, whichever is first.
    // A request's own deadline can also be earlier than the batch deadline.
    TimePoint wake = deadline;
    for (const auto& r : queue_) {
      if (r->deadline && *r->deadline < wake) wake = *r->deadline;
    }
    cv_.wait_until(lock, wake);
  }
}

void RequestQueue::Close(bool drain) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    drain_ = drain;
  }
  cv_.notify_all();
}

std::vector<InferenceRequestPtr> RequestQueue::TakeAll() {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<InferenceRequestPtr> out;
  out.reserve(queue_.size());
  for (auto& r : queue_) out.push_back(std::move(r));
  queue_.clear();
  return out;
}

size_t RequestQueue::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return queue_.size();
}

bool RequestQueue::closed() const {
  std::lock_guard<std::mutex> lock(mu_);
  return closed_;
}

}  // namespace mie
