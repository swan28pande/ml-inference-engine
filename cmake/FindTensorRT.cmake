# FindTensorRT
# ------------
# Locates NVIDIA TensorRT (nvinfer + nvonnxparser).
#
# Hints:   TENSORRT_ROOT (CMake or environment variable)
# Defines: TensorRT_FOUND, TensorRT_VERSION, imported targets
#          TensorRT::nvinfer and TensorRT::nvonnxparser

set(_trt_hints ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} /usr /usr/local/tensorrt /opt/tensorrt)

find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_trt_hints}
  PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)

find_library(TensorRT_nvinfer_LIBRARY
  NAMES nvinfer nvinfer_10
  HINTS ${_trt_hints}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)

find_library(TensorRT_nvonnxparser_LIBRARY
  NAMES nvonnxparser nvonnxparser_10
  HINTS ${_trt_hints}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_lines
       REGEX "^#define (NV_TENSORRT_MAJOR|NV_TENSORRT_MINOR|NV_TENSORRT_PATCH|TRT_MAJOR_ENTERPRISE|TRT_MINOR_ENTERPRISE|TRT_PATCH_ENTERPRISE) [0-9]+")
  foreach(_line ${_trt_ver_lines})
    if(_line MATCHES "^#define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE) ([0-9]+)")
      set(_trt_major ${CMAKE_MATCH_2})
    elseif(_line MATCHES "^#define (NV_TENSORRT_MINOR|TRT_MINOR_ENTERPRISE) ([0-9]+)")
      set(_trt_minor ${CMAKE_MATCH_2})
    elseif(_line MATCHES "^#define (NV_TENSORRT_PATCH|TRT_PATCH_ENTERPRISE) ([0-9]+)")
      set(_trt_patch ${CMAKE_MATCH_2})
    endif()
  endforeach()
  set(TensorRT_VERSION "${_trt_major}.${_trt_minor}.${_trt_patch}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY TensorRT_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
      IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  endif()
  if(NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
      IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
  endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY)
