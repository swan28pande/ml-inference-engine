#include "mie/engine/session_manager.h"

#include <algorithm>
#include <cctype>
#include <mutex>

#include "mie/common/logging.h"

namespace mie {
namespace {

bool IsInteger(const std::string& s) {
  return !s.empty() && s.size() < 19 &&
         std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

}  // namespace

bool SessionManager::VersionLess(const std::string& a, const std::string& b) {
  if (IsInteger(a) && IsInteger(b)) return std::stoll(a) < std::stoll(b);
  return a < b;
}

SessionManager::SessionManager(std::shared_ptr<MetricsRegistry> metrics)
    : metrics_(metrics ? std::move(metrics) : std::make_shared<MetricsRegistry>()) {}

SessionManager::~SessionManager() { UnloadAll(/*drain=*/false); }

Status SessionManager::LoadModel(const ModelConfig& config) {
  {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = models_.find(config.name);
    if (it != models_.end() && it->second.count(config.version)) {
      return AlreadyExists("model " + config.name + ":" + config.version + " is already loaded");
    }
  }

  // Load outside the lock: building a TensorRT engine can take minutes.
  auto session = ModelSession::Create(config, metrics_->GetOrCreate(config.name, config.version));

  std::unique_lock<std::shared_mutex> lock(mu_);
  if (!session.ok()) {
    failures_[{config.name, config.version}] = session.status().message();
    MIE_LOG(ERROR) << session.status().ToString();
    return session.status();
  }
  auto& versions = models_[config.name];
  if (versions.count(config.version)) {
    lock.unlock();
    session.value()->Shutdown(false);
    return AlreadyExists("model " + config.name + ":" + config.version + " was loaded concurrently");
  }
  versions.emplace(config.version, std::move(session).value());
  failures_.erase({config.name, config.version});
  return Status::Ok();
}

Status SessionManager::UnloadModel(const std::string& name, const std::string& version, bool drain) {
  std::shared_ptr<ModelSession> session;
  {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = models_.find(name);
    if (it == models_.end()) return NotFound("model " + name + " is not loaded");
    auto vit = it->second.find(version);
    if (vit == it->second.end()) return NotFound("model " + name + ":" + version + " is not loaded");
    session = std::move(vit->second);
    it->second.erase(vit);
    if (it->second.empty()) models_.erase(it);
  }
  // Drain outside the lock so other models keep serving.
  session->Shutdown(drain);
  MIE_LOG(INFO) << "unloaded model " << name << ":" << version;
  return Status::Ok();
}

void SessionManager::UnloadAll(bool drain) {
  std::map<std::string, VersionMap> models;
  {
    std::unique_lock<std::shared_mutex> lock(mu_);
    models.swap(models_);
  }
  for (auto& [name, versions] : models) {
    for (auto& [version, session] : versions) session->Shutdown(drain);
  }
}

StatusOr<std::shared_ptr<ModelSession>> SessionManager::GetSession(const std::string& name,
                                                                   const std::string& version) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = models_.find(name);
  if (it == models_.end()) return NotFound("model '" + name + "' is not loaded");
  if (!version.empty()) {
    auto vit = it->second.find(version);
    if (vit == it->second.end()) return NotFound("model '" + name + "' has no version '" + version + "'");
    return vit->second;
  }
  std::shared_ptr<ModelSession> latest;
  std::string latest_version;
  for (const auto& [v, session] : it->second) {
    if (session->state() != ModelState::kReady) continue;
    if (!latest || VersionLess(latest_version, v)) {
      latest = session;
      latest_version = v;
    }
  }
  if (!latest) return Unavailable("model '" + name + "' has no READY version");
  return latest;
}

std::vector<std::string> SessionManager::ReadyVersions(const std::string& name) const {
  std::vector<std::string> out;
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = models_.find(name);
  if (it == models_.end()) return out;
  for (const auto& [v, session] : it->second) {
    if (session->state() == ModelState::kReady) out.push_back(v);
  }
  std::sort(out.begin(), out.end(), VersionLess);
  return out;
}

std::vector<std::string> SessionManager::ModelNames() const {
  std::vector<std::string> out;
  std::shared_lock<std::shared_mutex> lock(mu_);
  for (const auto& [name, versions] : models_) out.push_back(name);
  for (const auto& [key, msg] : failures_) {
    if (!models_.count(key.first) && std::find(out.begin(), out.end(), key.first) == out.end()) {
      out.push_back(key.first);
    }
  }
  return out;
}

std::vector<ModelVersionStatus> SessionManager::ListStatus() const {
  std::vector<ModelVersionStatus> out;
  std::shared_lock<std::shared_mutex> lock(mu_);
  for (const auto& [name, versions] : models_) {
    for (const auto& [version, session] : versions) {
      ModelVersionStatus s;
      s.model_name = name;
      s.version = version;
      s.backend = session->config().backend;
      s.state = session->state();
      s.queue_depth = session->queue_depth();
      out.push_back(std::move(s));
    }
  }
  for (const auto& [key, msg] : failures_) {
    ModelVersionStatus s;
    s.model_name = key.first;
    s.version = key.second;
    s.state = ModelState::kFailed;
    s.message = msg;
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace mie
