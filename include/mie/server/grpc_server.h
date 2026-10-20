// gRPC front-end: InferenceService (callback API) + standard grpc.health.v1.
//
// Predict never blocks a gRPC thread: the request is handed to the model's
// batch scheduler and the RPC is completed from the scheduler's callback.
//
// Graceful shutdown:
//   1. stop admitting new requests (UNAVAILABLE) and report NOT_SERVING /
//      DRAINING on both health endpoints so load balancers stop routing;
//   2. wait up to drain_timeout for in-flight requests to finish;
//   3. stop all model schedulers (failing anything still queued);
//   4. shut down the gRPC server.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "mie/engine/concurrency_limiter.h"
#include "mie/engine/session_manager.h"
#include "mie/server/config_loader.h"
#include "mie/server/health_monitor.h"
#include "mie/server/traffic_router.h"

namespace grpc {
class Server;
}  // namespace grpc

namespace mie {

class InferenceServiceImpl;

class InferenceServer {
 public:
  explicit InferenceServer(ServerOptions options);
  ~InferenceServer();

  InferenceServer(const InferenceServer&) = delete;
  InferenceServer& operator=(const InferenceServer&) = delete;

  // Loads all configured models, then starts listening. Fails if any model
  // fails to load.
  Status Start();

  // Graceful shutdown (idempotent, thread-safe).
  void Shutdown();

  // Blocks until the server has shut down.
  void Wait();

  // Port actually bound (useful when listening on port 0).
  int port() const { return port_; }

  SessionManager* sessions() { return sessions_.get(); }
  TrafficRouter* router() { return &router_; }
  HealthMonitor* health() { return health_.get(); }
  ConcurrencyLimiter* limiter() { return &limiter_; }
  bool draining() const { return draining_.load(); }

 private:
  ServerOptions options_;
  std::shared_ptr<MetricsRegistry> metrics_;
  std::unique_ptr<SessionManager> sessions_;
  TrafficRouter router_;
  ConcurrencyLimiter limiter_;
  std::unique_ptr<HealthMonitor> health_;
  std::unique_ptr<InferenceServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::atomic<bool> draining_{false};
  std::atomic<bool> shutdown_{false};
  std::mutex shutdown_mu_;
  int port_ = 0;
};

}  // namespace mie
