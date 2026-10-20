cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)  # honour ONNXRUNTIME_ROOT / TENSORRT_ROOT
endif()

project(ml_inference_engine
  VERSION 0.1.0
  DESCRIPTION "High-performance ML inference server (ONNX Runtime / TensorRT / TorchScript, dynamic batching, gRPC)"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(MIE_WITH_ONNXRUNTIME "Build the ONNX Runtime backend" ON)
option(MIE_WITH_CUDA        "Build CUDA support (device memory pool, kernels)" OFF)
option(MIE_WITH_TENSORRT    "Build the TensorRT backend (requires MIE_WITH_CUDA)" OFF)
option(MIE_WITH_TORCH       "Build the TorchScript (LibTorch) backend" OFF)
option(MIE_WITH_NVML        "Use NVML for GPU utilization reporting (requires MIE_WITH_CUDA)" ON)
option(MIE_BUILD_SERVER     "Build the gRPC server" ON)
option(MIE_BUILD_TESTS      "Build unit tests" ON)
option(MIE_BUILD_BENCHMARKS "Build benchmarks" ON)
set(MIE_SANITIZER "" CACHE STRING "Sanitizer to build with: address (ASan+UBSan) | thread (TSan) | empty")
option(MIE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(MIE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures to compile kernels for")

if(MIE_WITH_TENSORRT AND NOT MIE_WITH_CUDA)
  message(FATAL_ERROR "MIE_WITH_TENSORRT requires MIE_WITH_CUDA=ON")
endif()

include(CompilerOptions)
include(Dependencies)

# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
add_subdirectory(src)

if(MIE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MIE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "ml-inference-engine ${PROJECT_VERSION} configuration")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler    : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  ONNX Runtime    : ${MIE_WITH_ONNXRUNTIME}")
message(STATUS "  CUDA            : ${MIE_WITH_CUDA}")
message(STATUS "  TensorRT        : ${MIE_WITH_TENSORRT}")
message(STATUS "  TorchScript     : ${MIE_WITH_TORCH}")
message(STATUS "  NVML            : ${MIE_WITH_NVML}")
message(STATUS "  gRPC server     : ${MIE_BUILD_SERVER}")
message(STATUS "  Tests           : ${MIE_BUILD_TESTS}")
message(STATUS "  Benchmarks      : ${MIE_BUILD_BENCHMARKS}")
message(STATUS "  Sanitizer       : ${MIE_SANITIZER}")
message(STATUS "")
