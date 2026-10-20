#include "mie/server/traffic_router.h"

#include <algorithm>
#include <mutex>
#include <random>

namespace mie {

uint64_t TrafficRouter::HashKey(const std::string& key) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
  for (unsigned char c : key) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  // Final avalanche so short, similar keys spread evenly.
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return h;
}

Status TrafficRouter::SetSplit(const std::string& model, const TrafficWeights& weights) {
  uint64_t total = 0;
  for (const auto& [version, w] : weights) {
    if (version.empty()) return InvalidArgument("traffic split contains an empty version");
    total += w;
  }
  std::unique_lock<std::shared_mutex> lock(mu_);
  if (weights.empty()) {
    splits_.erase(model);
    return Status::Ok();
  }
  if (total == 0) return InvalidArgument("traffic split weights must not all be zero");
  splits_[model] = weights;
  return Status::Ok();
}

TrafficWeights TrafficRouter::GetSplit(const std::string& model) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = splits_.find(model);
  return it == splits_.end() ? TrafficWeights{} : it->second;
}

std::map<std::string, TrafficWeights> TrafficRouter::AllSplits() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return splits_;
}

StatusOr<std::string> TrafficRouter::Route(const std::string& model, const std::string& requested_version,
                                           const std::string& routing_key,
                                           const std::vector<std::string>& ready_versions) const {
  if (!requested_version.empty()) return requested_version;
  if (ready_versions.empty()) return Unavailable("model '" + model + "' has no READY version");

  std::vector<std::pair<std::string, uint32_t>> candidates;
  uint64_t total = 0;
  {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = splits_.find(model);
    if (it != splits_.end()) {
      for (const auto& [version, weight] : it->second) {
        if (weight == 0) continue;
        if (std::find(ready_versions.begin(), ready_versions.end(), version) == ready_versions.end()) continue;
        candidates.emplace_back(version, weight);
        total += weight;
      }
    }
  }
  if (candidates.empty()) return ready_versions.back();  // latest READY version
  if (candidates.size() == 1) return candidates.front().first;

  uint64_t point;
  if (!routing_key.empty()) {
    point = HashKey(routing_key) % total;
  } else {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    point = std::uniform_int_distribution<uint64_t>(0, total - 1)(rng);
  }
  uint64_t cumulative = 0;
  for (const auto& [version, weight] : candidates) {
    cumulative += weight;
    if (point < cumulative) return version;
  }
  return candidates.back().first;
}

}  // namespace mie
