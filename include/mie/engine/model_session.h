// A loaded, servable model version: backend instances + batch scheduler.
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "mie/engine/batch_scheduler.h"
#include "mie/engine/metrics.h"
#include "mie/engine/model_backend.h"

namespace mie {

enum class ModelState { kLoading, kReady, kUnloading, kFailed };
const char* ModelStateName(ModelState state);

class ModelSession {
 public:
  // Loads `config.instance_count` backend instances, runs warm-up and starts
  // the scheduler.
  static StatusOr<std::shared_ptr<ModelSession>> Create(const ModelConfig& config,
                                                        std::shared_ptr<ModelMetrics> metrics);
  ~ModelSession();

  ModelSession(const ModelSession&) = delete;
  ModelSession& operator=(const ModelSession&) = delete;

  // Asynchronous inference. The callback runs exactly once (synchronously on
  // rejection, otherwise on a scheduler thread).
  void Submit(InferenceRequestPtr request);

  // Convenience wrapper returning a future.
  std::future<InferenceResult> InferAsync(TensorMap inputs, std::vector<std::string> requested_outputs = {},
                                          std::optional<TimePoint> deadline = std::nullopt);
  InferenceResult Infer(TensorMap inputs);

  // Stops accepting requests, drains (or fails) queued ones and joins workers.
  void Shutdown(bool drain);

  const ModelConfig& config() const { return config_; }
  const ModelMetadata& metadata() const { return metadata_; }
  ModelState state() const { return state_.load(); }
  size_t queue_depth() const { return scheduler_ ? scheduler_->queue_depth() : 0; }
  ModelMetrics* metrics() const { return metrics_.get(); }

 private:
  ModelSession(ModelConfig config, std::shared_ptr<ModelMetrics> metrics);

  ModelConfig config_;
  ModelMetadata metadata_;
  std::shared_ptr<ModelMetrics> metrics_;
  std::vector<std::unique_ptr<ModelBackend>> instances_;
  std::unique_ptr<BatchScheduler> scheduler_;
  std::atomic<ModelState> state_{ModelState::kLoading};
  std::atomic<uint64_t> next_request_id_{1};
};

}  // namespace mie
