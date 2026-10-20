// Version routing for A/B tests and canary rollouts.
//
// Each model may have a weight split across versions, e.g. {"1": 90, "2": 10}.
// Route() resolves the version for a request:
//   1. an explicitly requested version always wins;
//   2. with a split, a non-empty routing key is hashed (FNV-1a) so the same
//      key always lands on the same version (sticky assignment); without a
//      key a weighted random choice is made;
//   3. versions that are not READY are excluded and their weight is
//      redistributed; with no usable split the latest READY version is used.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "mie/common/status.h"

namespace mie {

using TrafficWeights = std::map<std::string, uint32_t>;

class TrafficRouter {
 public:
  // Replaces the split for `model`. An empty map clears it.
  Status SetSplit(const std::string& model, const TrafficWeights& weights);
  TrafficWeights GetSplit(const std::string& model) const;
  std::map<std::string, TrafficWeights> AllSplits() const;

  // `ready_versions` must be sorted oldest -> newest.
  StatusOr<std::string> Route(const std::string& model, const std::string& requested_version,
                              const std::string& routing_key, const std::vector<std::string>& ready_versions) const;

  static uint64_t HashKey(const std::string& key);

 private:
  mutable std::shared_mutex mu_;
  std::map<std::string, TrafficWeights> splits_;
};

}  // namespace mie
