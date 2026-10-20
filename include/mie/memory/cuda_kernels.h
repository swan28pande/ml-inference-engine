// Device kernels used by GPU backends for on-device pre/post-processing.
// Only available when built with MIE_WITH_CUDA.
#pragma once

#ifdef MIE_WITH_CUDA

#include <cstdint>
#include <cuda_runtime_api.h>

namespace mie {
namespace cuda {

// float32 -> float16 conversion (for engines built with FP16 I/O).
cudaError_t CastFloatToHalf(const float* src, void* dst_half, int64_t count, cudaStream_t stream);

// float16 -> float32 conversion.
cudaError_t CastHalfToFloat(const void* src_half, float* dst, int64_t count, cudaStream_t stream);

// Converts a batch of interleaved uint8 HWC images to normalised float32 NCHW:
//   dst[n][c][y][x] = (src[n][y][x][c] / 255 - mean[c]) / stddev[c]
// `mean` and `stddev` hold `channels` values each (max 4 channels).
cudaError_t NormalizeHwcToNchw(const uint8_t* src, float* dst, int batch, int height, int width,
                               int channels, const float* mean, const float* stddev, cudaStream_t stream);

}  // namespace cuda
}  // namespace mie

#endif  // MIE_WITH_CUDA
