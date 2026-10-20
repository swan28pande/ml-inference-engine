#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mie/common/clock.h"
#include "mie/common/status.h"
#include "mie/common/tensor.h"

namespace mie {

struct RequestTiming {
  int64_t queue_us = 0;    // enqueue -> batch execution start
  int64_t compute_us = 0;  // backend execution time of the batch
  int64_t total_us = 0;    // enqueue -> completion
  int batch_size = 0;      // samples in the executed batch
};

struct InferenceResult {
  Status status;
  TensorMap outputs;
  RequestTiming timing;
};

using InferenceCallback = std::function<void(InferenceResult&&)>;

// A single client request travelling through the engine. The callback is
// invoked exactly once, from a scheduler worker thread (or synchronously from
// Submit() if the request is rejected).
struct InferenceRequest {
  uint64_t id = 0;
  TensorMap inputs;
  std::vector<std::string> requested_outputs;  // empty = all outputs
  std::optional<TimePoint> deadline;           // fail with DEADLINE_EXCEEDED if not started by then
  InferenceCallback callback;

  // Filled in by the scheduler.
  TimePoint enqueue_time{};
  int64_t batch_size = 0;         // dim 0 shared by all inputs
  std::string batch_signature;    // requests with equal signatures can be batched together
};

using InferenceRequestPtr = std::unique_ptr<InferenceRequest>;

// Validates `request` and fills batch_size / batch_signature.
Status PrepareRequest(InferenceRequest* request);

}  // namespace mie
