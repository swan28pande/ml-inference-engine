// Loads ServerConfig (protobuf text format) into engine-level options.
#pragma once

#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "mie/common/status.h"
#include "mie/engine/model_config.h"
#include "mie/memory/memory_pool.h"
#include "mie/server/traffic_router.h"

namespace mie {

struct ServerOptions {
  std::string listen_address = "0.0.0.0:50051";
  size_t max_concurrent_requests = 8192;  // 0 = unlimited
  std::chrono::milliseconds drain_timeout{10000};
  int max_message_bytes = 64 << 20;
  MemoryPoolOptions device_pool;
  std::vector<std::pair<size_t, size_t>> device_pool_preallocate;  // (bytes, count)
  std::vector<ModelConfig> models;
  std::map<std::string, TrafficWeights> traffic;
};

// Parses a text-format ServerConfig. Relative model paths are resolved
// against `base_dir`.
StatusOr<ServerOptions> ParseServerConfig(const std::string& text, const std::string& base_dir);
StatusOr<ServerOptions> LoadServerConfig(const std::string& path);

}  // namespace mie
