#include "mie/server/config_loader.h"

#include <google/protobuf/text_format.h>

#include <fstream>
#include <sstream>

#include "server_config.pb.h"

namespace mie {
namespace {

std::string JoinPath(const std::string& base, const std::string& path) {
  if (path.empty() || path.front() == '/' || base.empty()) return path;
  return base + "/" + path;
}

std::string DirName(const std::string& path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? "." : path.substr(0, pos);
}

StatusOr<TensorSpec> ConvertSpec(const config::TensorSpecConfig& c) {
  TensorSpec spec;
  spec.name = c.name();
  spec.dtype = DataTypeFromString(c.dtype().empty() ? "float32" : c.dtype());
  if (spec.dtype == DataType::kInvalid) return InvalidArgument("unknown dtype '" + c.dtype() + "' for " + c.name());
  spec.shape.assign(c.shape().begin(), c.shape().end());
  return spec;
}

}  // namespace

StatusOr<ServerOptions> ParseServerConfig(const std::string& text, const std::string& base_dir) {
  config::ServerConfig cfg;
  if (!google::protobuf::TextFormat::ParseFromString(text, &cfg)) {
    return InvalidArgument("failed to parse server config (protobuf text format)");
  }

  ServerOptions opts;
  if (!cfg.listen_address().empty()) opts.listen_address = cfg.listen_address();
  if (cfg.max_concurrent_requests()) opts.max_concurrent_requests = cfg.max_concurrent_requests();
  if (cfg.drain_timeout_ms()) opts.drain_timeout = std::chrono::milliseconds(cfg.drain_timeout_ms());
  if (cfg.max_message_bytes()) opts.max_message_bytes = static_cast<int>(cfg.max_message_bytes());

  if (cfg.has_device_memory_pool()) {
    const auto& p = cfg.device_memory_pool();
    if (p.max_cached_bytes()) opts.device_pool.max_cached_bytes = p.max_cached_bytes();
    if (p.min_block_bytes()) opts.device_pool.min_block_bytes = p.min_block_bytes();
    if (p.max_block_bytes()) opts.device_pool.max_block_bytes = p.max_block_bytes();
    for (const auto& pre : p.preallocate()) opts.device_pool_preallocate.emplace_back(pre.bytes(), pre.count());
  }

  for (const auto& m : cfg.models()) {
    if (m.name().empty()) return InvalidArgument("every model needs a name");
    if (m.backend().empty()) return InvalidArgument("model '" + m.name() + "' needs a backend");
    ModelConfig mc;
    mc.name = m.name();
    mc.version = m.version().empty() ? "1" : m.version();
    mc.backend = m.backend();
    mc.path = JoinPath(base_dir, m.path());
    mc.instance_count = m.instance_count() ? static_cast<int>(m.instance_count()) : 1;
    mc.use_gpu = m.use_gpu();
    mc.device_id = m.device_id();
    mc.warmup_iterations = static_cast<int>(m.warmup_iterations());
    for (const auto& [k, v] : m.parameters()) mc.parameters[k] = v;
    if (m.has_batching()) {
      const auto& b = m.batching();
      mc.batching.enabled = !b.disabled();
      if (b.max_batch_size()) mc.batching.max_batch_size = b.max_batch_size();
      mc.batching.max_queue_delay_us = b.max_queue_delay_us();
      mc.batching.preferred_batch_sizes.assign(b.preferred_batch_sizes().begin(), b.preferred_batch_sizes().end());
      if (b.max_queue_size()) mc.batching.max_queue_size = b.max_queue_size();
    }
    for (const auto& in : m.inputs()) {
      auto spec = ConvertSpec(in);
      if (!spec.ok()) return spec.status();
      mc.inputs.push_back(std::move(spec).value());
    }
    for (const auto& out : m.outputs()) {
      auto spec = ConvertSpec(out);
      if (!spec.ok()) return spec.status();
      mc.outputs.push_back(std::move(spec).value());
    }
    opts.models.push_back(std::move(mc));
  }

  for (const auto& t : cfg.traffic()) {
    TrafficWeights w(t.weights().begin(), t.weights().end());
    opts.traffic[t.model_name()] = std::move(w);
  }
  return opts;
}

StatusOr<ServerOptions> LoadServerConfig(const std::string& path) {
  std::ifstream in(path);
  if (!in) return NotFound("cannot open config file " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ParseServerConfig(ss.str(), DirName(path));
}

}  // namespace mie
