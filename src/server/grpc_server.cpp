#include "mie/server/grpc_server.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <algorithm>

#include "inference.grpc.pb.h"
#include "mie/common/logging.h"
#include "mie/server/proto_convert.h"

#ifndef MIE_VERSION
#define MIE_VERSION "0.1.0"
#endif

namespace mie {
namespace {

v1::ModelState ToProto(ModelState s) {
  switch (s) {
    case ModelState::kLoading: return v1::LOADING;
    case ModelState::kReady: return v1::READY;
    case ModelState::kUnloading: return v1::UNLOADING;
    case ModelState::kFailed: return v1::FAILED;
  }
  return v1::MODEL_STATE_UNKNOWN;
}

void ToProto(const ModelVersionStatus& s, v1::ModelStatus* p) {
  p->set_model_name(s.model_name);
  p->set_version(s.version);
  p->set_state(ToProto(s.state));
  p->set_backend(s.backend);
  p->set_queue_depth(s.queue_depth);
  p->set_message(s.message);
}

void ToProto(const LatencyHistogram::Summary& s, v1::LatencySummary* p) {
  p->set_count(s.count);
  p->set_mean_us(s.mean);
  p->set_p50_us(s.p50);
  p->set_p95_us(s.p95);
  p->set_p99_us(s.p99);
  p->set_max_us(static_cast<double>(s.max));
}

// Converts the client's absolute deadline (system clock) to the steady clock.
std::optional<TimePoint> SteadyDeadline(const grpc::CallbackServerContext* ctx) {
  const auto deadline = ctx->deadline();
  if (deadline >= std::chrono::system_clock::time_point::max() - std::chrono::hours(24 * 365)) return std::nullopt;
  return Clock::now() + std::chrono::duration_cast<Clock::duration>(deadline - std::chrono::system_clock::now());
}

}  // namespace

class InferenceServiceImpl final : public v1::InferenceService::CallbackService {
 public:
  explicit InferenceServiceImpl(InferenceServer* server) : server_(server) {}

  grpc::ServerUnaryReactor* Predict(grpc::CallbackServerContext* ctx, const v1::PredictRequest* req,
                                    v1::PredictResponse* resp) override {
    auto* reactor = ctx->DefaultReactor();
    if (server_->draining()) {
      reactor->Finish(grpc::Status(grpc::StatusCode::UNAVAILABLE, "server is shutting down"));
      return reactor;
    }
    auto permit = server_->limiter()->TryAcquire();
    if (!permit) {
      reactor->Finish(grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                   "too many concurrent requests (limit " +
                                       std::to_string(server_->limiter()->limit()) + ")"));
      return reactor;
    }

    SessionManager* sessions = server_->sessions();
    const auto ready = sessions->ReadyVersions(req->model_name());
    if (ready.empty() && req->model_version().empty()) {
      reactor->Finish(grpc::Status(grpc::StatusCode::NOT_FOUND, "model '" + req->model_name() + "' is not loaded"));
      return reactor;
    }
    auto version = server_->router()->Route(req->model_name(), req->model_version(), req->routing_key(), ready);
    if (!version.ok()) {
      reactor->Finish(ToGrpcStatus(version.status()));
      return reactor;
    }
    auto session = sessions->GetSession(req->model_name(), version.value());
    if (!session.ok()) {
      reactor->Finish(ToGrpcStatus(session.status()));
      return reactor;
    }

    auto request = std::make_unique<InferenceRequest>();
    for (const auto& t : req->inputs()) {
      Tensor tensor;
      Status s = FromProto(t, &tensor);
      if (!s.ok()) {
        reactor->Finish(ToGrpcStatus(s));
        return reactor;
      }
      request->inputs.emplace(t.name(), std::move(tensor));
    }
    request->requested_outputs.assign(req->output_filter().begin(), req->output_filter().end());
    request->deadline = SteadyDeadline(ctx);

    // The permit is released when the callback (and the request) is destroyed.
    auto held = std::make_shared<ConcurrencyLimiter::Permit>(std::move(permit));
    const TimePoint start = Clock::now();
    request->callback = [reactor, resp, held, start, name = req->model_name(),
                         ver = version.value()](InferenceResult&& result) {
      if (!result.status.ok()) {
        reactor->Finish(ToGrpcStatus(result.status));
        return;
      }
      resp->set_model_name(name);
      resp->set_model_version(ver);
      for (const auto& [out_name, tensor] : result.outputs) ToProto(out_name, tensor, resp->add_outputs());
      auto* timing = resp->mutable_timing();
      timing->set_queue_us(result.timing.queue_us);
      timing->set_compute_us(result.timing.compute_us);
      timing->set_batch_size(result.timing.batch_size);
      timing->set_total_us(MicrosSince(start));
      reactor->Finish(grpc::Status::OK);
    };
    session.value()->Submit(std::move(request));
    return reactor;
  }

  grpc::ServerUnaryReactor* Health(grpc::CallbackServerContext* ctx, const v1::HealthRequest* /*req*/,
                                   v1::HealthResponse* resp) override {
    const HealthReport report = server_->health()->Report();
    switch (report.state) {
      case ServingState::kServing: resp->set_status(v1::SERVING); break;
      case ServingState::kNotServing: resp->set_status(v1::NOT_SERVING); break;
      case ServingState::kDraining: resp->set_status(v1::DRAINING); break;
    }
    for (const auto& m : report.models) ToProto(m, resp->add_models());
    for (const auto& g : report.gpus) {
      auto* gp = resp->add_gpus();
      gp->set_device_id(g.device_id);
      gp->set_name(g.name);
      gp->set_utilization_percent(g.utilization_percent);
      gp->set_memory_used_bytes(g.memory_used_bytes);
      gp->set_memory_total_bytes(g.memory_total_bytes);
    }
    resp->set_uptime_seconds(report.uptime_seconds);
    resp->set_in_flight_requests(server_->limiter()->in_flight());
    resp->set_server_version(MIE_VERSION);
    auto* reactor = ctx->DefaultReactor();
    reactor->Finish(grpc::Status::OK);
    return reactor;
  }

  grpc::ServerUnaryReactor* GetModelMetadata(grpc::CallbackServerContext* ctx, const v1::ModelMetadataRequest* req,
                                             v1::ModelMetadataResponse* resp) override {
    auto* reactor = ctx->DefaultReactor();
    auto session = server_->sessions()->GetSession(req->model_name(), req->model_version());
    if (!session.ok()) {
      reactor->Finish(ToGrpcStatus(session.status()));
      return reactor;
    }
    const ModelMetadata& md = session.value()->metadata();
    resp->set_model_name(md.name);
    resp->set_version(md.version);
    resp->set_backend(md.backend);
    resp->set_platform(md.platform);
    resp->set_max_batch_size(session.value()->config().batching.enabled
                                 ? std::min(md.max_batch_size, session.value()->config().batching.max_batch_size)
                                 : 1);
    for (const auto& in : md.inputs) mie::ToProto(in, resp->add_inputs());
    for (const auto& out : md.outputs) mie::ToProto(out, resp->add_outputs());
    for (const auto& [k, v] : md.properties) (*resp->mutable_properties())[k] = v;
    reactor->Finish(grpc::Status::OK);
    return reactor;
  }

  grpc::ServerUnaryReactor* ListModels(grpc::CallbackServerContext* ctx, const v1::ListModelsRequest* /*req*/,
                                       v1::ListModelsResponse* resp) override {
    const auto statuses = server_->sessions()->ListStatus();
    std::map<std::string, v1::ModelInfo*> by_name;
    for (const auto& s : statuses) {
      auto*& info = by_name[s.model_name];
      if (!info) {
        info = resp->add_models();
        info->set_model_name(s.model_name);
        for (const auto& [v, w] : server_->router()->GetSplit(s.model_name)) (*info->mutable_traffic_weights())[v] = w;
      }
      ToProto(s, info->add_versions());
    }
    auto* reactor = ctx->DefaultReactor();
    reactor->Finish(grpc::Status::OK);
    return reactor;
  }

  grpc::ServerUnaryReactor* SetTrafficSplit(grpc::CallbackServerContext* ctx, const v1::SetTrafficSplitRequest* req,
                                            v1::SetTrafficSplitResponse* resp) override {
    auto* reactor = ctx->DefaultReactor();
    TrafficWeights weights(req->weights().begin(), req->weights().end());
    for (const auto& [version, w] : weights) {
      (void)w;
      auto s = server_->sessions()->GetSession(req->model_name(), version);
      if (!s.ok()) {
        reactor->Finish(ToGrpcStatus(s.status()));
        return reactor;
      }
    }
    Status s = server_->router()->SetSplit(req->model_name(), weights);
    if (!s.ok()) {
      reactor->Finish(ToGrpcStatus(s));
      return reactor;
    }
    for (const auto& [v, w] : server_->router()->GetSplit(req->model_name())) (*resp->mutable_weights())[v] = w;
    MIE_LOG(INFO) << "traffic split for " << req->model_name() << " updated (" << weights.size() << " versions)";
    reactor->Finish(grpc::Status::OK);
    return reactor;
  }

  grpc::ServerUnaryReactor* GetMetrics(grpc::CallbackServerContext* ctx, const v1::MetricsRequest* req,
                                       v1::MetricsResponse* resp) override {
    MetricsRegistry* registry = server_->sessions()->metrics();
    for (const auto& s : registry->SnapshotAll()) {
      if (!req->model_name().empty() && req->model_name() != s.model_name) continue;
      auto* m = resp->add_models();
      m->set_model_name(s.model_name);
      m->set_version(s.version);
      m->set_requests_ok(s.requests_ok);
      m->set_requests_failed(s.requests_failed);
      m->set_requests_rejected(s.requests_rejected);
      m->set_throughput_rps(s.throughput_rps);
      m->set_mean_batch_size(s.mean_batch_size);
      ToProto(s.request_latency, m->mutable_request_latency());
      ToProto(s.queue_latency, m->mutable_queue_latency());
      ToProto(s.compute_latency, m->mutable_compute_latency());
    }
    if (req->prometheus_format()) resp->set_prometheus_text(registry->RenderPrometheus());
    auto* reactor = ctx->DefaultReactor();
    reactor->Finish(grpc::Status::OK);
    return reactor;
  }

 private:
  InferenceServer* server_;
};

// ------------------------------------------------------------- InferenceServer

InferenceServer::InferenceServer(ServerOptions options)
    : options_(std::move(options)),
      metrics_(std::make_shared<MetricsRegistry>()),
      sessions_(std::make_unique<SessionManager>(metrics_)),
      limiter_(options_.max_concurrent_requests),
      health_(std::make_unique<HealthMonitor>(sessions_.get())) {}

InferenceServer::~InferenceServer() { Shutdown(); }

Status InferenceServer::Start() {
  DeviceMemoryPools::Instance().Configure(options_.device_pool);
#ifdef MIE_WITH_CUDA
  if (!options_.device_pool_preallocate.empty()) {
    auto pool = DeviceMemoryPools::Instance().Get(0);
    if (pool.ok()) {
      for (const auto& [bytes, count] : options_.device_pool_preallocate) {
        Status s = pool.value()->Reserve(bytes, count);
        if (!s.ok()) MIE_LOG(WARNING) << "device pool pre-allocation failed: " << s.ToString();
      }
    }
  }
#endif

  for (const auto& model : options_.models) MIE_RETURN_IF_ERROR(sessions_->LoadModel(model));
  for (const auto& [model, weights] : options_.traffic) MIE_RETURN_IF_ERROR(router_.SetSplit(model, weights));

  service_ = std::make_unique<InferenceServiceImpl>(this);
  grpc::EnableDefaultHealthCheckService(true);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(options_.listen_address, grpc::InsecureServerCredentials(), &port_);
  builder.SetMaxReceiveMessageSize(options_.max_message_bytes);
  builder.SetMaxSendMessageSize(options_.max_message_bytes);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  if (!server_ || port_ == 0) return Unavailable("failed to listen on " + options_.listen_address);

  if (auto* hc = server_->GetHealthCheckService()) hc->SetServingStatus(true);
  health_->Start();
  health_->SetState(ServingState::kServing);
  MIE_LOG(INFO) << "mie_server " << MIE_VERSION << " listening on port " << port_ << " with "
                << options_.models.size() << " model version(s)";
  return Status::Ok();
}

void InferenceServer::Shutdown() {
  std::lock_guard<std::mutex> lock(shutdown_mu_);
  if (shutdown_.exchange(true)) return;
  if (!server_) {
    sessions_->UnloadAll(false);
    return;
  }

  // 1) Stop admitting work and tell load balancers.
  draining_.store(true);
  health_->SetState(ServingState::kDraining);
  if (auto* hc = server_->GetHealthCheckService()) hc->SetServingStatus(false);
  MIE_LOG(INFO) << "graceful shutdown: draining " << limiter_.in_flight() << " in-flight request(s) (timeout "
                << options_.drain_timeout.count() << " ms)";

  // 2) Wait for in-flight requests.
  const bool drained = limiter_.WaitForIdle(options_.drain_timeout);
  if (!drained) {
    MIE_LOG(WARNING) << "drain timeout expired with " << limiter_.in_flight() << " request(s) still in flight";
  }

  // 3) Stop schedulers; anything still queued completes with UNAVAILABLE so
  //    every outstanding reactor is finished before the server shuts down.
  sessions_->UnloadAll(/*drain=*/drained);

  // 4) Close the transport.
  server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
  health_->Stop();
  health_->SetState(ServingState::kNotServing);
  MIE_LOG(INFO) << "shutdown complete";
}

void InferenceServer::Wait() {
  if (server_) server_->Wait();
}

}  // namespace mie
