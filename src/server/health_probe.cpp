// mie_health_probe: tiny readiness probe for Docker / Kubernetes.
//
//   mie_health_probe [--address localhost:50051] [--timeout_ms 2000]
//
// Exits 0 if the server reports SERVING and at least one model is READY.
#include <grpcpp/grpcpp.h>

#include <chrono>
#include <iostream>
#include <string>

#include "inference.grpc.pb.h"

int main(int argc, char** argv) {
  std::string address = "localhost:50051";
  int timeout_ms = 2000;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--address") address = argv[i + 1];
    else if (arg == "--timeout_ms") timeout_ms = std::stoi(argv[i + 1]);
  }

  auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = mie::v1::InferenceService::NewStub(channel);
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
  mie::v1::HealthRequest req;
  mie::v1::HealthResponse resp;
  const grpc::Status status = stub->Health(&ctx, req, &resp);
  if (!status.ok()) {
    std::cerr << "health check failed: " << status.error_message() << "\n";
    return 1;
  }
  bool any_ready = false;
  for (const auto& m : resp.models()) any_ready = any_ready || m.state() == mie::v1::READY;
  std::cout << mie::v1::ServingStatus_Name(resp.status()) << " models=" << resp.models_size()
            << " in_flight=" << resp.in_flight_requests() << "\n";
  return (resp.status() == mie::v1::SERVING && any_ready) ? 0 : 1;
}
