#include <gtest/gtest.h>

#include "mie/server/config_loader.h"

namespace mie {
namespace {

TEST(ConfigLoaderTest, ParsesFullConfig) {
  const char* text = R"pb(
    listen_address: "127.0.0.1:6000"
    max_concurrent_requests: 100
    drain_timeout_ms: 1500
    device_memory_pool { max_cached_bytes: 1048576 preallocate { bytes: 4096 count: 4 } }
    models {
      name: "m"
      version: "3"
      backend: "reference"
      path: "models/m.onnx"
      instance_count: 2
      batching { max_batch_size: 16 max_queue_delay_us: 250 preferred_batch_sizes: [4, 8] }
      parameters { key: "input_dim" value: "8" }
      inputs { name: "input" dtype: "float32" shape: [-1, 8] }
    }
    traffic { model_name: "m" weights { key: "3" value: 100 } }
  )pb";
  auto opts = ParseServerConfig(text, "/srv");
  ASSERT_TRUE(opts.ok()) << opts.status().ToString();
  EXPECT_EQ(opts->listen_address, "127.0.0.1:6000");
  EXPECT_EQ(opts->max_concurrent_requests, 100u);
  EXPECT_EQ(opts->drain_timeout.count(), 1500);
  EXPECT_EQ(opts->device_pool.max_cached_bytes, 1048576u);
  ASSERT_EQ(opts->device_pool_preallocate.size(), 1u);
  ASSERT_EQ(opts->models.size(), 1u);
  const ModelConfig& m = opts->models[0];
  EXPECT_EQ(m.path, "/srv/models/m.onnx");
  EXPECT_EQ(m.instance_count, 2);
  EXPECT_TRUE(m.batching.enabled);
  EXPECT_EQ(m.batching.max_batch_size, 16);
  EXPECT_EQ(m.batching.max_queue_delay_us, 250);
  EXPECT_EQ(m.batching.preferred_batch_sizes, (std::vector<int>{4, 8}));
  EXPECT_EQ(m.GetIntParameter("input_dim", 0), 8);
  ASSERT_EQ(m.inputs.size(), 1u);
  EXPECT_EQ(m.inputs[0].shape, (Shape{-1, 8}));
  EXPECT_EQ(opts->traffic.at("m").at("3"), 100u);
}

TEST(ConfigLoaderTest, RejectsInvalidConfigs) {
  EXPECT_FALSE(ParseServerConfig("this is not a config", ".").ok());
  EXPECT_FALSE(ParseServerConfig("models { backend: \"reference\" }", ".").ok());
  EXPECT_FALSE(ParseServerConfig("models { name: \"m\" }", ".").ok());
  EXPECT_FALSE(
      ParseServerConfig("models { name: \"m\" backend: \"reference\" inputs { name: \"x\" dtype: \"complex\" } }", ".")
          .ok());
  EXPECT_EQ(LoadServerConfig("/does/not/exist.pbtxt").status().code(), StatusCode::kNotFound);
}

TEST(ConfigLoaderTest, ShippedConfigsParse) {
  for (const char* name : {"server.pbtxt", "server.reference.pbtxt", "server.gpu.pbtxt"}) {
    auto opts = LoadServerConfig(std::string(MIE_TEST_CONFIGS_DIR) + "/" + name);
    EXPECT_TRUE(opts.ok()) << name << ": " << opts.status().ToString();
    if (opts.ok()) EXPECT_FALSE(opts->models.empty()) << name;
  }
}

}  // namespace
}  // namespace mie
