#include "mie/engine/model_session.h"

#include <algorithm>
#include <exception>

#include "mie/common/logging.h"

namespace mie {

const char* ModelStateName(ModelState state) {
  switch (state) {
    case ModelState::kLoading: return "LOADING";
    case ModelState::kReady: return "READY";
    case ModelState::kUnloading: return "UNLOADING";
    case ModelState::kFailed: return "FAILED";
  }
  return "UNKNOWN";
}

ModelSession::ModelSession(ModelConfig config, std::shared_ptr<ModelMetrics> metrics)
    : config_(std::move(config)), metrics_(std::move(metrics)) {}

ModelSession::~ModelSession() { Shutdown(/*drain=*/false); }

StatusOr<std::shared_ptr<ModelSession>> ModelSession::Create(const ModelConfig& config,
                                                             std::shared_ptr<ModelMetrics> metrics) {
  if (config.name.empty()) return InvalidArgument("model config has no name");
  if (!metrics) metrics = std::make_shared<ModelMetrics>(config.name, config.version);
  std::shared_ptr<ModelSession> session(new ModelSession(config, std::move(metrics)));

  const int instances = std::max(1, config.instance_count);
  const TimePoint start = Clock::now();
  for (int i = 0; i < instances; ++i) {
    auto backend = CreateBackend(config.backend);
    if (!backend.ok()) return backend.status();
    Status s;
    try {
      s = backend.value()->Load(config);
    } catch (const std::exception& e) {
      s = Internal(std::string("exception while loading model: ") + e.what());
    }
    if (!s.ok()) {
      return Status(s.code(), "failed to load " + config.name + ":" + config.version + " (" + config.backend +
                                  "): " + s.message());
    }
    session->instances_.push_back(std::move(backend).value());
  }

  session->metadata_ = session->instances_.front()->metadata();
  session->metadata_.name = config.name;
  session->metadata_.version = config.version;
  session->metadata_.backend = config.backend;

  if (config.warmup_iterations > 0) {
    const int max_batch = std::max(1, std::min(config.batching.max_batch_size,
                                               std::max(1, session->metadata_.max_batch_size)));
    for (auto& instance : session->instances_) {
      for (int batch : {1, max_batch}) {
        Status s;
        try {
          s = instance->Warmup(batch, config.warmup_iterations);
        } catch (const std::exception& e) {
          s = Internal(e.what());
        }
        if (!s.ok()) return Status(s.code(), "warm-up failed for " + config.name + ": " + s.message());
      }
    }
  }

  std::vector<ModelBackend*> ptrs;
  for (auto& b : session->instances_) ptrs.push_back(b.get());
  session->scheduler_ = std::make_unique<BatchScheduler>(config.batching, session->metadata_.max_batch_size,
                                                         std::move(ptrs), session->metrics_);
  session->scheduler_->Start();
  session->state_.store(ModelState::kReady);

  const auto& policy = session->scheduler_->policy();
  MIE_LOG(INFO) << "loaded model " << config.name << ":" << config.version << " backend=" << config.backend
                << " platform=" << session->metadata_.platform << " instances=" << instances
                << " max_batch=" << policy.max_batch_size << " max_delay_us=" << policy.max_queue_delay_us
                << " in " << MicrosSince(start) / 1000 << " ms";
  return session;
}

void ModelSession::Submit(InferenceRequestPtr request) {
  auto reject = [&](Status status) {
    if (status.code() == StatusCode::kResourceExhausted) {
      metrics_->RecordRejected();
    } else {
      metrics_->RecordFailure();
    }
    InferenceResult result;
    result.status = std::move(status);
    if (request->callback) {
      InferenceCallback cb = std::move(request->callback);
      cb(std::move(result));
    }
  };

  if (state_.load(std::memory_order_acquire) != ModelState::kReady) {
    reject(Unavailable("model " + config_.name + ":" + config_.version + " is " + ModelStateName(state())));
    return;
  }
  request->id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  Status s = scheduler_->Enqueue(request);
  if (!s.ok()) reject(std::move(s));
}

std::future<InferenceResult> ModelSession::InferAsync(TensorMap inputs, std::vector<std::string> requested_outputs,
                                                      std::optional<TimePoint> deadline) {
  auto promise = std::make_shared<std::promise<InferenceResult>>();
  auto future = promise->get_future();
  auto request = std::make_unique<InferenceRequest>();
  request->inputs = std::move(inputs);
  request->requested_outputs = std::move(requested_outputs);
  request->deadline = deadline;
  request->callback = [promise](InferenceResult&& result) { promise->set_value(std::move(result)); };
  Submit(std::move(request));
  return future;
}

InferenceResult ModelSession::Infer(TensorMap inputs) { return InferAsync(std::move(inputs)).get(); }

void ModelSession::Shutdown(bool drain) {
  if (state_.exchange(ModelState::kUnloading) == ModelState::kUnloading) return;
  if (scheduler_) scheduler_->Stop(drain);
}

}  // namespace mie
