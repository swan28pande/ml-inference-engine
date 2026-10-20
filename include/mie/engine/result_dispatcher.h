// Maps the outputs of an executed batch back to the requests that formed it.
#pragma once

#include <vector>

#include "mie/common/status.h"
#include "mie/engine/inference_request.h"
#include "mie/engine/metrics.h"

namespace mie {

struct BatchExecution {
  TimePoint start;         // when the batch started executing
  int64_t compute_us = 0;  // backend time
  int batch_size = 0;      // total samples
};

class ResultDispatcher {
 public:
  explicit ResultDispatcher(ModelMetrics* metrics) : metrics_(metrics) {}

  // Splits every output along dim 0 according to each request's batch size,
  // applies per-request output filters and invokes callbacks. If `status` is
  // an error, every request in the batch fails with it.
  void Dispatch(std::vector<InferenceRequestPtr>& batch, const Status& status, TensorMap& outputs,
                const BatchExecution& exec) const;

  // Fails a single request without executing it.
  void Fail(InferenceRequest& request, const Status& status) const;

 private:
  void Complete(InferenceRequest& request, InferenceResult&& result) const;
  ModelMetrics* metrics_;
};

}  // namespace mie
