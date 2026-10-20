include(GoogleTest)

set(MIE_TEST_SOURCES
  test_tensor.cpp
  test_memory_pool.cpp
  test_request_queue.cpp
  test_batch_scheduler.cpp
  test_metrics.cpp
  test_model_loading.cpp)

if(MIE_BUILD_SERVER)
  list(APPEND MIE_TEST_SOURCES
    test_traffic_router.cpp
    test_config_loader.cpp
    test_grpc_server.cpp)
endif()

add_executable(mie_tests ${MIE_TEST_SOURCES})
if(MIE_BUILD_SERVER)
  target_link_libraries(mie_tests PRIVATE mie::server GTest::gtest_main mie::compile_options)
else()
  target_link_libraries(mie_tests PRIVATE mie::core GTest::gtest_main mie::compile_options)
endif()
target_compile_definitions(mie_tests PRIVATE
  MIE_TEST_MODELS_DIR="${PROJECT_SOURCE_DIR}/models/sample"
  MIE_TEST_CONFIGS_DIR="${PROJECT_SOURCE_DIR}/configs")

gtest_discover_tests(mie_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 120)
