// Bounded request queue with batch formation.
//
// PopBatch() implements the dynamic batching policy:
//   * a batch is dispatched immediately once compatible requests add up to
//     max_batch_size samples;
//   * otherwise it is dispatched when the oldest request has waited
//     max_queue_delay_us (so a lone request pays at most that much latency);
//   * only requests with identical input signatures (names, dtypes, non-batch
//     dims) are grouped; others stay queued in arrival order;
//   * requests whose deadline already passed are removed and returned
//     separately so they can be failed without executing.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

#include "mie/engine/inference_request.h"
#include "mie/engine/model_config.h"

namespace mie {

struct BatchPolicy {
  int max_batch_size = 1;
  int64_t max_queue_delay_us = 0;
  std::vector<int> preferred_batch_sizes;  // sorted ascending
};

class RequestQueue {
 public:
  explicit RequestQueue(size_t capacity);

  // Non-blocking. Fails with RESOURCE_EXHAUSTED when full or UNAVAILABLE when
  // closed; on failure the request is left untouched in `request`.
  Status Push(InferenceRequestPtr& request);

  // Blocks until a batch is ready per `policy`. Expired requests are moved to
  // `expired`. Returns false once the queue is closed and fully drained.
  bool PopBatch(const BatchPolicy& policy, std::vector<InferenceRequestPtr>* batch,
                std::vector<InferenceRequestPtr>* expired);

  // Stops accepting new requests. With drain=true, PopBatch keeps returning
  // queued work (without waiting for delays) until empty; with drain=false the
  // remaining requests are returned by TakeAll().
  void Close(bool drain);
  std::vector<InferenceRequestPtr> TakeAll();

  size_t size() const;
  size_t capacity() const { return capacity_; }
  bool closed() const;

 private:
  // Selects requests compatible with the head into `batch`. Requires mu_.
  void TakeCompatible(const BatchPolicy& policy, bool timed_out, std::vector<InferenceRequestPtr>* batch);
  void RemoveExpired(TimePoint now, std::vector<InferenceRequestPtr>* expired);
  int64_t CompatibleSamples(int max_batch_size) const;

  const size_t capacity_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<InferenceRequestPtr> queue_;
  bool closed_ = false;
  bool drain_ = false;
};

}  // namespace mie
