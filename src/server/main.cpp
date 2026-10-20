// mie_server: gRPC inference server entry point.
//
//   mie_server --config configs/server.pbtxt [--address 0.0.0.0:50051] [--log_level info]
//
// SIGINT / SIGTERM trigger a graceful shutdown (drain, then exit).
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <pthread.h>

#include "mie/common/logging.h"
#include "mie/engine/model_backend.h"
#include "mie/server/config_loader.h"
#include "mie/server/grpc_server.h"

namespace {

void PrintUsage() {
  std::cout << "Usage: mie_server --config <server.pbtxt> [options]\n"
               "\n"
               "Options:\n"
               "  --config PATH        server configuration (protobuf text format)\n"
               "  --address HOST:PORT  override listen_address from the config\n"
               "  --log_level LEVEL    debug | info | warning | error (default: info)\n"
               "  --list_backends      print backends compiled into this binary and exit\n"
               "  --help               show this message\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  std::string address_override;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--config") {
      config_path = value();
    } else if (arg == "--address") {
      address_override = value();
    } else if (arg == "--log_level") {
      mie::LogLevel level;
      const std::string v = value();
      if (!mie::ParseLogLevel(v.c_str(), &level)) {
        std::cerr << "invalid log level: " << v << "\n";
        return 2;
      }
      mie::SetLogLevel(level);
    } else if (arg == "--list_backends") {
      for (const auto& b : mie::AvailableBackends()) std::cout << b << "\n";
      return 0;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      PrintUsage();
      return 2;
    }
  }
  if (config_path.empty()) {
    PrintUsage();
    return 2;
  }

  // Block termination signals in every thread; the main thread waits for them
  // synchronously with sigwait(), which avoids async-signal-safety pitfalls.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  std::signal(SIGPIPE, SIG_IGN);

  auto options = mie::LoadServerConfig(config_path);
  if (!options.ok()) {
    MIE_LOG(ERROR) << options.status().ToString();
    return 1;
  }
  if (!address_override.empty()) options->listen_address = address_override;

  mie::InferenceServer server(std::move(options).value());
  mie::Status status = server.Start();
  if (!status.ok()) {
    MIE_LOG(ERROR) << "failed to start server: " << status.ToString();
    server.Shutdown();
    return 1;
  }

  int sig = 0;
  sigwait(&signals, &sig);
  MIE_LOG(INFO) << "received " << (sig == SIGTERM ? "SIGTERM" : "SIGINT");
  server.Shutdown();
  return 0;
}
