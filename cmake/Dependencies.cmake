# Resolves all third-party dependencies. Every optional dependency maps to a
# MIE_WITH_* option and a matching compile definition so that GPU-only code is
# compiled out cleanly in CPU-only builds (e.g. CI).

include(FetchContent)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------- ONNX Runtime
if(MIE_WITH_ONNXRUNTIME)
  find_package(ONNXRuntime)
  if(NOT ONNXRuntime_FOUND)
    message(WARNING
      "ONNX Runtime not found; the onnxruntime backend is disabled. "
      "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime (or run scripts/fetch_onnxruntime.sh) to enable it.")
    set(MIE_WITH_ONNXRUNTIME OFF CACHE BOOL "Build the ONNX Runtime backend" FORCE)
  endif()
endif()

# ------------------------------------------------------------------------ CUDA
if(MIE_WITH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${MIE_CUDA_ARCHITECTURES})
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  if(MIE_WITH_NVML AND NOT TARGET CUDA::nvml)
    message(WARNING "NVML not found in the CUDA toolkit; GPU utilization reporting falls back to cudaMemGetInfo.")
    set(MIE_WITH_NVML OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(MIE_WITH_NVML OFF CACHE BOOL "Use NVML for GPU utilization reporting (requires MIE_WITH_CUDA)" FORCE)
endif()

# -------------------------------------------------------------------- TensorRT
if(MIE_WITH_TENSORRT)
  find_package(TensorRT REQUIRED)
endif()

# ------------------------------------------------------------------- LibTorch
if(MIE_WITH_TORCH)
  find_package(Torch REQUIRED)
endif()

# --------------------------------------------------------------- gRPC/Protobuf
if(MIE_BUILD_SERVER)
  include(GrpcHelpers)
endif()

# ------------------------------------------------------------------ GoogleTest
if(MIE_BUILD_TESTS)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::gtest_main)
    message(STATUS "GoogleTest not found on the system; fetching v1.15.2")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()
