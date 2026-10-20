// Dynamic batch scheduler for one model version.
//
// One worker thread per backend instance pulls batches from a shared
// RequestQueue, so while one instance executes a batch the next batch is
// already being formed. Inputs are concatenated along dim 0, executed once,
// and the ResultDispatcher scatters outputs back to the original requests.
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "mie/engine/model_backend.h"
#include "mie/engine/request_queue.h"
#include "mie/engine/result_dispatcher.h"

namespace mie {

class BatchScheduler {
 public:
  // `instances` must outlive the scheduler. `max_model_batch` is the largest
  // batch the backend accepts (0 = no batching support).
  BatchScheduler(const BatchingConfig& config, int max_model_batch, std::vector<ModelBackend*> instances,
                 std::shared_ptr<ModelMetrics> metrics);
  ~BatchScheduler();

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  void Start();

  // Stops the workers. With drain=true queued requests are executed first;
  // otherwise they fail with UNAVAILABLE. Every callback is invoked before
  // Stop() returns.
  void Stop(bool drain);

  // Validates and enqueues a request. On error the request (and its
  // callback) is left in `request` and the caller owns error reporting.
  Status Enqueue(InferenceRequestPtr& request);

  size_t queue_depth() const { return queue_.size(); }
  const BatchPolicy& policy() const { return policy_; }

 private:
  void WorkerLoop(ModelBackend* backend);
  void ExecuteBatch(ModelBackend* backend, std::vector<InferenceRequestPtr>& batch);

  BatchPolicy policy_;
  std::vector<ModelBackend*> instances_;
  std::shared_ptr<ModelMetrics> metrics_;
  ResultDispatcher dispatcher_;
  RequestQueue queue_;
  std::vector<std::thread> workers_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stopped_{false};
  bool passthrough_ = false;  // backend has no dynamic batch dimension
};

}  // namespace mie
