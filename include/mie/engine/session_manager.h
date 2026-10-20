// Owns all loaded model versions. Lookups take a shared lock and return a
// shared_ptr, so a version being unloaded stays alive until in-flight
// requests holding it complete.
#pragma once

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "mie/engine/metrics.h"
#include "mie/engine/model_session.h"

namespace mie {

struct ModelVersionStatus {
  std::string model_name;
  std::string version;
  std::string backend;
  ModelState state = ModelState::kLoading;
  size_t queue_depth = 0;
  std::string message;
};

class SessionManager {
 public:
  explicit SessionManager(std::shared_ptr<MetricsRegistry> metrics);
  ~SessionManager();

  // Loads a model version. Fails with ALREADY_EXISTS if it is loaded.
  Status LoadModel(const ModelConfig& config);

  // Removes a version from routing, then drains and destroys it.
  Status UnloadModel(const std::string& name, const std::string& version, bool drain = true);

  // Shuts down every model (used by graceful shutdown).
  void UnloadAll(bool drain);

  // Returns the requested version, or the latest READY version if `version` is empty.
  StatusOr<std::shared_ptr<ModelSession>> GetSession(const std::string& name, const std::string& version = "") const;

  // READY versions of a model, sorted oldest -> newest.
  std::vector<std::string> ReadyVersions(const std::string& name) const;
  std::vector<std::string> ModelNames() const;
  std::vector<ModelVersionStatus> ListStatus() const;
  MetricsRegistry* metrics() const { return metrics_.get(); }

  // Orders versions numerically when both are integers, else lexicographically.
  static bool VersionLess(const std::string& a, const std::string& b);

 private:
  using VersionMap = std::map<std::string, std::shared_ptr<ModelSession>>;

  std::shared_ptr<MetricsRegistry> metrics_;
  mutable std::shared_mutex mu_;
  std::map<std::string, VersionMap> models_;
  std::map<std::pair<std::string, std::string>, std::string> failures_;  // load errors for reporting
};

}  // namespace mie
