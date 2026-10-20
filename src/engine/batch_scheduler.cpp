#include "mie/engine/batch_scheduler.h"

#include <algorithm>
#include <exception>

#include "mie/common/logging.h"

namespace mie {

BatchScheduler::BatchScheduler(const BatchingConfig& config, int max_model_batch,
                               std::vector<ModelBackend*> instances, std::shared_ptr<ModelMetrics> metrics)
    : instances_(std::move(instances)),
      metrics_(std::move(metrics)),
      dispatcher_(metrics_.get()),
      queue_(config.max_queue_size) {
  passthrough_ = max_model_batch <= 0;
  int max_batch = config.enabled ? config.max_batch_size : 1;
  if (passthrough_) {
    max_batch = 1;
  } else {
    max_batch = std::min(max_batch, max_model_batch);
  }
  policy_.max_batch_size = std::max(1, max_batch);
  policy_.max_queue_delay_us = policy_.max_batch_size > 1 ? std::max<int64_t>(0, config.max_queue_delay_us) : 0;
  for (int p : config.preferred_batch_sizes) {
    if (p > 0 && p <= policy_.max_batch_size) policy_.preferred_batch_sizes.push_back(p);
  }
  std::sort(policy_.preferred_batch_sizes.begin(), policy_.preferred_batch_sizes.end());
  policy_.preferred_batch_sizes.erase(
      std::unique(policy_.preferred_batch_sizes.begin(), policy_.preferred_batch_sizes.end()),
      policy_.preferred_batch_sizes.end());
}

BatchScheduler::~BatchScheduler() { Stop(/*drain=*/false); }

void BatchScheduler::Start() {
  if (started_.exchange(true)) return;
  workers_.reserve(instances_.size());
  for (ModelBackend* backend : instances_) {
    workers_.emplace_back([this, backend] { WorkerLoop(backend); });
  }
}

void BatchScheduler::Stop(bool drain) {
  if (stopped_.exchange(true)) return;
  queue_.Close(drain);
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  workers_.clear();
  // Anything left (drain=false, or never started) fails fast.
  for (auto& r : queue_.TakeAll()) dispatcher_.Fail(*r, Unavailable("model is shutting down"));
}

Status BatchScheduler::Enqueue(InferenceRequestPtr& request) {
  if (stopped_.load(std::memory_order_acquire)) return Unavailable("model is shutting down");
  MIE_RETURN_IF_ERROR(PrepareRequest(request.get()));
  if (passthrough_) {
    // The model has a fixed batch dimension: never merge requests.
    request->batch_signature += "#" + std::to_string(request->id);
  } else if (request->batch_size > policy_.max_batch_size) {
    return InvalidArgument("request batch size " + std::to_string(request->batch_size) +
                           " exceeds max_batch_size " + std::to_string(policy_.max_batch_size));
  }
  request->enqueue_time = Clock::now();
  return queue_.Push(request);
}

void BatchScheduler::WorkerLoop(ModelBackend* backend) {
  std::vector<InferenceRequestPtr> batch;
  std::vector<InferenceRequestPtr> expired;
  while (queue_.PopBatch(policy_, &batch, &expired)) {
    for (auto& r : expired) dispatcher_.Fail(*r, DeadlineExceeded("request deadline expired while queued"));
    if (!batch.empty()) ExecuteBatch(backend, batch);
  }
}

void BatchScheduler::ExecuteBatch(ModelBackend* backend, std::vector<InferenceRequestPtr>& batch) {
  BatchExecution exec;
  exec.start = Clock::now();
  for (const auto& r : batch) exec.batch_size += static_cast<int>(r->batch_size);

  // Gather: concatenate each input along the batch dimension.
  TensorMap gathered;
  const TensorMap* inputs = &batch.front()->inputs;
  Status status;
  if (batch.size() > 1) {
    std::vector<const Tensor*> parts(batch.size());
    for (const auto& [name, first] : batch.front()->inputs) {
      (void)first;
      for (size_t i = 0; i < batch.size(); ++i) parts[i] = &batch[i]->inputs.at(name);
      Tensor merged;
      status = ConcatBatch(parts, &merged);
      if (!status.ok()) break;
      gathered.emplace(name, std::move(merged));
    }
    inputs = &gathered;
  }

  TensorMap outputs;
  if (status.ok()) {
    const TimePoint compute_start = Clock::now();
    try {
      status = backend->Infer(*inputs, &outputs);
    } catch (const std::exception& e) {
      status = Internal(std::string("backend threw: ") + e.what());
    } catch (...) {
      status = Internal("backend threw an unknown exception");
    }
    exec.compute_us = MicrosSince(compute_start);
    if (status.ok() && metrics_) metrics_->RecordBatch(exec.batch_size, exec.compute_us);
  }
  if (!status.ok()) {
    MIE_LOG(WARNING) << "batch of " << batch.size() << " requests failed: " << status.ToString();
  }

  // Scatter: route each slice of the outputs back to its request.
  dispatcher_.Dispatch(batch, status, outputs, exec);
}

}  // namespace mie
