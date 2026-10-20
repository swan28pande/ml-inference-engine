// CUDA kernels for GPU-side tensor conversion and image pre-processing.
// Compiled only when MIE_WITH_CUDA=ON (see src/CMakeLists.txt).
#include "mie/memory/cuda_kernels.h"

#include <cuda_fp16.h>

namespace mie {
namespace cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxChannels = 4;

struct ChannelParams {
  float mean[kMaxChannels];
  float inv_std[kMaxChannels];
};

inline int BlocksFor(int64_t n) {
  const int64_t blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
  // Grid-stride loops below handle anything beyond this cap.
  return static_cast<int>(blocks < 65535 ? blocks : 65535);
}

__global__ void CastFloatToHalfKernel(const float* __restrict__ src, __half* __restrict__ dst, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    dst[i] = __float2half(src[i]);
  }
}

__global__ void CastHalfToFloatKernel(const __half* __restrict__ src, float* __restrict__ dst, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    dst[i] = __half2float(src[i]);
  }
}

// One thread per output element; reads are strided by `channels`, writes are
// fully coalesced along the x dimension of each output plane.
__global__ void NormalizeHwcToNchwKernel(const uint8_t* __restrict__ src, float* __restrict__ dst, int batch,
                                         int height, int width, int channels, ChannelParams params) {
  const int64_t plane = static_cast<int64_t>(height) * width;
  const int64_t total = static_cast<int64_t>(batch) * channels * plane;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t pixel = idx % plane;
    const int c = static_cast<int>((idx / plane) % channels);
    const int64_t n = idx / (plane * channels);
    const uint8_t v = src[(n * plane + pixel) * channels + c];
    dst[idx] = (static_cast<float>(v) * (1.0f / 255.0f) - params.mean[c]) * params.inv_std[c];
  }
}

}  // namespace

cudaError_t CastFloatToHalf(const float* src, void* dst_half, int64_t count, cudaStream_t stream) {
  if (count <= 0) return cudaSuccess;
  CastFloatToHalfKernel<<<BlocksFor(count), kThreadsPerBlock, 0, stream>>>(src, static_cast<__half*>(dst_half),
                                                                           count);
  return cudaGetLastError();
}

cudaError_t CastHalfToFloat(const void* src_half, float* dst, int64_t count, cudaStream_t stream) {
  if (count <= 0) return cudaSuccess;
  CastHalfToFloatKernel<<<BlocksFor(count), kThreadsPerBlock, 0, stream>>>(
      static_cast<const __half*>(src_half), dst, count);
  return cudaGetLastError();
}

cudaError_t NormalizeHwcToNchw(const uint8_t* src, float* dst, int batch, int height, int width, int channels,
                               const float* mean, const float* stddev, cudaStream_t stream) {
  if (channels <= 0 || channels > kMaxChannels) return cudaErrorInvalidValue;
  ChannelParams params{};
  for (int c = 0; c < channels; ++c) {
    params.mean[c] = mean[c];
    params.inv_std[c] = stddev[c] != 0.0f ? 1.0f / stddev[c] : 1.0f;
  }
  const int64_t total = static_cast<int64_t>(batch) * channels * height * width;
  if (total <= 0) return cudaSuccess;
  NormalizeHwcToNchwKernel<<<BlocksFor(total), kThreadsPerBlock, 0, stream>>>(src, dst, batch, height, width,
                                                                              channels, params);
  return cudaGetLastError();
}

}  // namespace cuda
}  // namespace mie
