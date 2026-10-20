#include "mie/server/health_monitor.h"

#include "mie/common/logging.h"
#include "mie/memory/cuda_utils.h"

#ifdef MIE_WITH_NVML
#include <nvml.h>
#endif

namespace mie {

#ifdef MIE_WITH_NVML
namespace {
bool InitNvml() {
  static const bool ok = [] {
    const nvmlReturn_t r = nvmlInit_v2();
    if (r != NVML_SUCCESS) MIE_LOG(WARNING) << "NVML init failed: " << nvmlErrorString(r);
    return r == NVML_SUCCESS;
  }();
  return ok;
}
}  // namespace
#endif

std::vector<GpuStatus> QueryGpus() {
  std::vector<GpuStatus> out;
#ifdef MIE_WITH_NVML
  if (InitNvml()) {
    unsigned int count = 0;
    if (nvmlDeviceGetCount_v2(&count) == NVML_SUCCESS) {
      for (unsigned int i = 0; i < count; ++i) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS) continue;
        GpuStatus g;
        g.device_id = static_cast<int>(i);
        char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE] = {0};
        if (nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS) g.name = name;
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) g.utilization_percent = util.gpu;
        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
          g.memory_used_bytes = mem.used;
          g.memory_total_bytes = mem.total;
        }
        out.push_back(std::move(g));
      }
      return out;
    }
  }
#endif
#ifdef MIE_WITH_CUDA
  for (int i = 0; i < CudaDeviceCount(); ++i) {
    auto info = GetCudaDeviceInfo(i);
    if (!info.ok()) continue;
    GpuStatus g;
    g.device_id = i;
    g.name = info->name;
    g.memory_total_bytes = info->total_memory;
    g.memory_used_bytes = info->total_memory - info->free_memory;
    out.push_back(std::move(g));
  }
#endif
  return out;
}

HealthMonitor::HealthMonitor(const SessionManager* sessions, std::chrono::milliseconds sample_interval)
    : sessions_(sessions), interval_(sample_interval), start_time_(Clock::now()) {}

HealthMonitor::~HealthMonitor() { Stop(); }

void HealthMonitor::Start() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    gpus_ = QueryGpus();
    stop_ = false;
  }
  if (!sampler_.joinable()) sampler_ = std::thread([this] { SampleLoop(); });
}

void HealthMonitor::Stop() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (sampler_.joinable()) sampler_.join();
}

void HealthMonitor::SampleLoop() {
  std::unique_lock<std::mutex> lock(mu_);
  while (!stop_) {
    if (cv_.wait_for(lock, interval_, [&] { return stop_; })) break;
    lock.unlock();
    auto gpus = QueryGpus();
    lock.lock();
    gpus_ = std::move(gpus);
  }
}

HealthReport HealthMonitor::Report() const {
  HealthReport r;
  r.state = state_.load();
  r.models = sessions_->ListStatus();
  {
    std::lock_guard<std::mutex> lock(mu_);
    r.gpus = gpus_;
  }
  r.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start_time_).count();
  return r;
}

}  // namespace mie
