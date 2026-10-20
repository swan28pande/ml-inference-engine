#include "mie/engine/result_dispatcher.h"

#include <exception>

#include "mie/common/logging.h"

namespace mie {
namespace {

// Applies the request's output filter. Returns an error if a requested output
// does not exist.
Status FilterOutputs(const std::vector<std::string>& requested, TensorMap* outputs) {
  if (requested.empty()) return Status::Ok();
  TensorMap filtered;
  for (const auto& name : requested) {
    auto it = outputs->find(name);
    if (it == outputs->end()) return InvalidArgument("model has no output named '" + name + "'");
    filtered.emplace(name, std::move(it->second));
  }
  *outputs = std::move(filtered);
  return Status::Ok();
}

}  // namespace

void ResultDispatcher::Complete(InferenceRequest& request, InferenceResult&& result) const {
  result.timing.total_us = MicrosSince(request.enqueue_time);
  if (metrics_) {
    if (result.status.ok()) {
      metrics_->RecordSuccess(result.timing.total_us, result.timing.queue_us);
    } else {
      metrics_->RecordFailure();
    }
  }
  if (!request.callback) return;
  // Move the callback out so it cannot be invoked twice.
  InferenceCallback cb = std::move(request.callback);
  request.callback = nullptr;
  try {
    cb(std::move(result));
  } catch (const std::exception& e) {
    MIE_LOG(ERROR) << "inference callback for request " << request.id << " threw: " << e.what();
  } catch (...) {
    MIE_LOG(ERROR) << "inference callback for request " << request.id << " threw an unknown exception";
  }
}

void ResultDispatcher::Fail(InferenceRequest& request, const Status& status) const {
  InferenceResult result;
  result.status = status;
  Complete(request, std::move(result));
}

void ResultDispatcher::Dispatch(std::vector<InferenceRequestPtr>& batch, const Status& status, TensorMap& outputs,
                                const BatchExecution& exec) const {
  if (!status.ok()) {
    for (auto& r : batch) Fail(*r, status);
    return;
  }

  auto timing_for = [&](const InferenceRequest& r) {
    RequestTiming t;
    t.queue_us = MicrosBetween(r.enqueue_time, exec.start);
    t.compute_us = exec.compute_us;
    t.batch_size = exec.batch_size;
    return t;
  };

  // Fast path: no scatter needed.
  if (batch.size() == 1) {
    InferenceRequest& r = *batch.front();
    InferenceResult result;
    result.timing = timing_for(r);
    result.status = FilterOutputs(r.requested_outputs, &outputs);
    if (result.status.ok()) result.outputs = std::move(outputs);
    Complete(r, std::move(result));
    return;
  }

  std::vector<int64_t> sizes;
  sizes.reserve(batch.size());
  for (const auto& r : batch) sizes.push_back(r->batch_size);

  std::vector<TensorMap> per_request(batch.size());
  std::vector<Tensor> pieces;
  for (auto& [name, tensor] : outputs) {
    Status s = SplitBatch(tensor, sizes, &pieces);
    if (!s.ok()) {
      const Status err = Internal("cannot scatter output '" + name + "' back to requests: " + s.message());
      for (auto& r : batch) Fail(*r, err);
      return;
    }
    for (size_t i = 0; i < batch.size(); ++i) per_request[i].emplace(name, std::move(pieces[i]));
  }

  for (size_t i = 0; i < batch.size(); ++i) {
    InferenceRequest& r = *batch[i];
    InferenceResult result;
    result.timing = timing_for(r);
    result.status = FilterOutputs(r.requested_outputs, &per_request[i]);
    if (result.status.ok()) result.outputs = std::move(per_request[i]);
    Complete(r, std::move(result));
  }
}

}  // namespace mie
