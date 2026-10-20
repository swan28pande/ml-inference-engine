// Aggregates server health: serving state, per-model status and GPU stats.
//
// GPU utilisation is sampled by a background thread (NVML when available,
// otherwise memory usage via cudaMemGetInfo) so the Health RPC never blocks on
// driver calls.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mie/common/clock.h"
#include "mie/engine/session_manager.h"

namespace mie {

struct GpuStatus {
  int device_id = 0;
  std::string name;
  double utilization_percent = -1;  // -1 when unknown
  uint64_t memory_used_bytes = 0;
  uint64_t memory_total_bytes = 0;
};

enum class ServingState { kServing, kNotServing, kDraining };

struct HealthReport {
  ServingState state = ServingState::kNotServing;
  std::vector<ModelVersionStatus> models;
  std::vector<GpuStatus> gpus;
  int64_t uptime_seconds = 0;
};

// Queries GPU status directly (may take a few milliseconds).
std::vector<GpuStatus> QueryGpus();

class HealthMonitor {
 public:
  explicit HealthMonitor(const SessionManager* sessions, std::chrono::milliseconds sample_interval =
                                                             std::chrono::milliseconds(1000));
  ~HealthMonitor();

  void Start();
  void Stop();

  void SetState(ServingState state) { state_.store(state); }
  ServingState state() const { return state_.load(); }

  HealthReport Report() const;

 private:
  void SampleLoop();

  const SessionManager* sessions_;
  const std::chrono::milliseconds interval_;
  const TimePoint start_time_;
  std::atomic<ServingState> state_{ServingState::kNotServing};

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<GpuStatus> gpus_;
  bool stop_ = false;
  std::thread sampler_;
};

}  // namespace mie
