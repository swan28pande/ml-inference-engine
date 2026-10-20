// CUDA helpers. Everything in this header compiles in CPU-only builds; the
// CUDA-specific parts are guarded by MIE_WITH_CUDA.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "mie/common/status.h"

#ifdef MIE_WITH_CUDA
#include <cuda_runtime_api.h>

#define MIE_CUDA_RETURN_IF_ERROR(expr)                                                      \
  do {                                                                                      \
    cudaError_t _mie_err = (expr);                                                          \
    if (_mie_err != cudaSuccess) {                                                          \
      return ::mie::Internal(std::string(#expr) + " failed: " + cudaGetErrorString(_mie_err)); \
    }                                                                                       \
  } while (0)
#endif

namespace mie {

struct CudaDeviceInfo {
  int device_id = 0;
  std::string name;
  size_t total_memory = 0;
  size_t free_memory = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int multiprocessor_count = 0;
};

// True when the binary was built with CUDA and at least one device is visible.
bool CudaAvailable();
int CudaDeviceCount();
StatusOr<CudaDeviceInfo> GetCudaDeviceInfo(int device_id);

#ifdef MIE_WITH_CUDA
// RAII wrapper over a non-blocking CUDA stream.
class CudaStream {
 public:
  explicit CudaStream(int device_id);
  ~CudaStream();
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  cudaStream_t get() const { return stream_; }
  Status Synchronize() const;

 private:
  cudaStream_t stream_ = nullptr;
};
#endif

}  // namespace mie
