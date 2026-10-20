# --------------------------------------------------------------------- mie_core
# Engine library: tensors, memory pools, backends, batching, metrics. Has no
# gRPC/protobuf dependency so it can be embedded in other services.

add_library(mie_core
  common/logging.cpp
  common/status.cpp
  common/tensor.cpp
  memory/cuda_utils.cpp
  memory/memory_pool.cpp
  engine/batch_scheduler.cpp
  engine/inference_request.cpp
  engine/metrics.cpp
  engine/model_backend.cpp
  engine/model_session.cpp
  engine/request_queue.cpp
  engine/result_dispatcher.cpp
  engine/session_manager.cpp
  engine/backends/reference_backend.cpp
  engine/backends/onnxruntime_backend.cpp
  engine/backends/tensorrt_backend.cpp
  engine/backends/torchscript_backend.cpp)
add_library(mie::core ALIAS mie_core)

target_include_directories(mie_core PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mie_core
  PUBLIC Threads::Threads
  PRIVATE mie::compile_options)

if(MIE_WITH_ONNXRUNTIME)
  target_compile_definitions(mie_core PUBLIC MIE_WITH_ONNXRUNTIME=1)
  target_link_libraries(mie_core PUBLIC ONNXRuntime::ONNXRuntime)
  if(ONNXRuntime_LIBRARY_DIR)
    set_target_properties(mie_core PROPERTIES INTERFACE_LINK_DIRECTORIES "${ONNXRuntime_LIBRARY_DIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${ONNXRuntime_LIBRARY_DIR}" PARENT_SCOPE)
  endif()
endif()

if(MIE_WITH_CUDA)
  target_sources(mie_core PRIVATE memory/cuda_kernels.cu)
  target_compile_definitions(mie_core PUBLIC MIE_WITH_CUDA=1)
  target_link_libraries(mie_core PUBLIC CUDA::cudart)
  set_target_properties(mie_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  if(MIE_WITH_NVML)
    target_compile_definitions(mie_core PUBLIC MIE_WITH_NVML=1)
    target_link_libraries(mie_core PUBLIC CUDA::nvml)
  endif()
endif()

if(MIE_WITH_TENSORRT)
  target_compile_definitions(mie_core PUBLIC MIE_WITH_TENSORRT=1)
  target_link_libraries(mie_core PUBLIC TensorRT::nvinfer TensorRT::nvonnxparser)
endif()

if(MIE_WITH_TORCH)
  target_compile_definitions(mie_core PUBLIC MIE_WITH_TORCH=1)
  target_link_libraries(mie_core PUBLIC ${TORCH_LIBRARIES})
endif()

# ------------------------------------------------------------------- gRPC server
if(MIE_BUILD_SERVER)
  mie_generate_grpc(MIE_PROTO_SRCS ${PROJECT_SOURCE_DIR}/proto
    ${PROJECT_SOURCE_DIR}/proto/inference.proto
    ${PROJECT_SOURCE_DIR}/proto/server_config.proto)

  add_library(mie_proto ${MIE_PROTO_SRCS})
  add_library(mie::proto ALIAS mie_proto)
  target_include_directories(mie_proto PUBLIC ${CMAKE_BINARY_DIR}/generated)
  target_link_libraries(mie_proto PUBLIC ${MIE_GRPCPP_TARGET} ${MIE_PROTOBUF_TARGET})

  add_library(mie_server_lib
    server/config_loader.cpp
    server/grpc_server.cpp
    server/health_monitor.cpp
    server/proto_convert.cpp
    server/traffic_router.cpp)
  add_library(mie::server ALIAS mie_server_lib)
  target_link_libraries(mie_server_lib
    PUBLIC mie::core mie::proto
    PRIVATE mie::compile_options)

  add_executable(mie_server server/main.cpp)
  target_link_libraries(mie_server PRIVATE mie::server mie::compile_options)

  add_executable(mie_health_probe server/health_probe.cpp)
  target_link_libraries(mie_health_probe PRIVATE mie::proto mie::compile_options)

  install(TARGETS mie_server mie_health_probe RUNTIME DESTINATION bin)
endif()
