#include "mie/memory/cuda_utils.h"

#include "mie/common/logging.h"

namespace mie {

int CudaDeviceCount() {
#ifdef MIE_WITH_CUDA
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return count;
#else
  return 0;
#endif
}

bool CudaAvailable() { return CudaDeviceCount() > 0; }

StatusOr<CudaDeviceInfo> GetCudaDeviceInfo(int device_id) {
#ifdef MIE_WITH_CUDA
  if (device_id < 0 || device_id >= CudaDeviceCount()) {
    return NotFound("CUDA device " + std::to_string(device_id) + " not found");
  }
  cudaDeviceProp prop{};
  MIE_CUDA_RETURN_IF_ERROR(cudaGetDeviceProperties(&prop, device_id));
  CudaDeviceInfo info;
  info.device_id = device_id;
  info.name = prop.name;
  info.compute_capability_major = prop.major;
  info.compute_capability_minor = prop.minor;
  info.multiprocessor_count = prop.multiProcessorCount;

  int previous = 0;
  MIE_CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous));
  MIE_CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
  size_t free_bytes = 0, total_bytes = 0;
  const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  cudaSetDevice(previous);
  MIE_CUDA_RETURN_IF_ERROR(err);
  info.free_memory = free_bytes;
  info.total_memory = total_bytes;
  return info;
#else
  (void)device_id;
  return Unavailable("built without CUDA support");
#endif
}

#ifdef MIE_WITH_CUDA
CudaStream::CudaStream(int device_id) {
  int previous = 0;
  cudaGetDevice(&previous);
  cudaSetDevice(device_id);
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
    MIE_LOG(ERROR) << "failed to create CUDA stream on device " << device_id;
    stream_ = nullptr;
  }
  cudaSetDevice(previous);
}

CudaStream::~CudaStream() {
  if (stream_) cudaStreamDestroy(stream_);
}

Status CudaStream::Synchronize() const {
  MIE_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
  return Status::Ok();
}
#endif

}  // namespace mie
