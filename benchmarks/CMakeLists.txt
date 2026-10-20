add_executable(mie_engine_benchmark engine_benchmark.cpp benchmark_util.h)
target_link_libraries(mie_engine_benchmark PRIVATE mie::core mie::compile_options)

if(MIE_BUILD_SERVER)
  add_executable(mie_loadgen grpc_load_test.cpp benchmark_util.h)
  target_link_libraries(mie_loadgen PRIVATE mie::core mie::proto mie::compile_options)
endif()
