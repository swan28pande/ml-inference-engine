#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <future>

#include "mie/engine/model_backend.h"
#include "mie/engine/session_manager.h"

namespace mie {
namespace {

ModelConfig Reference(const std::string& version, int input_dim = 4) {
  ModelConfig c;
  c.name = "ref";
  c.version = version;
  c.backend = "reference";
  c.parameters["input_dim"] = std::to_string(input_dim);
  c.parameters["output_dim"] = "2";
  c.parameters["seed"] = version;
  return c;
}

TEST(BackendFactoryTest, ReferenceAlwaysAvailable) {
  const auto backends = AvailableBackends();
  EXPECT_NE(std::find(backends.begin(), backends.end(), "reference"), backends.end());
  EXPECT_TRUE(CreateBackend("reference").ok());
  auto missing = CreateBackend("does-not-exist");
  EXPECT_EQ(missing.status().code(), StatusCode::kUnimplemented);
}

TEST(BackendFactoryTest, GpuBackendsCompiledOutWithoutCuda) {
#if !(defined(MIE_WITH_TENSORRT) && defined(MIE_WITH_CUDA))
  EXPECT_EQ(CreateBackend("tensorrt").status().code(), StatusCode::kUnimplemented);
#endif
#ifndef MIE_WITH_TORCH
  EXPECT_EQ(CreateBackend("torchscript").status().code(), StatusCode::kUnimplemented);
#endif
}

TEST(ReferenceBackendTest, LoadsAndReportsMetadata) {
  auto backend = CreateBackend("reference");
  ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend.value()->Load(Reference("1", 6)).ok());
  const auto& md = backend.value()->metadata();
  ASSERT_EQ(md.inputs.size(), 1u);
  EXPECT_EQ(md.inputs[0].name, "input");
  EXPECT_EQ(md.inputs[0].shape, (Shape{-1, 6}));
  EXPECT_GT(md.max_batch_size, 0);
  EXPECT_TRUE(backend.value()->Warmup(4, 2).ok());
}

TEST(ReferenceBackendTest, RejectsBadInputs) {
  auto backend = CreateBackend("reference");
  ASSERT_TRUE(backend.value()->Load(Reference("1", 4)).ok());
  TensorMap out;
  TensorMap wrong_name{{"x", Tensor(DataType::kFloat32, {1, 4})}};
  EXPECT_FALSE(backend.value()->Infer(wrong_name, &out).ok());
  TensorMap wrong_dim{{"input", Tensor(DataType::kFloat32, {1, 5})}};
  EXPECT_FALSE(backend.value()->Infer(wrong_dim, &out).ok());
  TensorMap wrong_type{{"input", Tensor(DataType::kInt64, {1, 4})}};
  EXPECT_FALSE(backend.value()->Infer(wrong_type, &out).ok());
}

TEST(SessionManagerTest, LoadsMultipleVersionsAndResolvesLatest) {
  SessionManager manager(nullptr);
  ASSERT_TRUE(manager.LoadModel(Reference("1")).ok());
  ASSERT_TRUE(manager.LoadModel(Reference("2")).ok());
  ASSERT_TRUE(manager.LoadModel(Reference("10")).ok());
  EXPECT_EQ(manager.LoadModel(Reference("2")).code(), StatusCode::kAlreadyExists);

  EXPECT_EQ(manager.ReadyVersions("ref"), (std::vector<std::string>{"1", "2", "10"}));  // numeric order
  auto latest = manager.GetSession("ref");
  ASSERT_TRUE(latest.ok());
  EXPECT_EQ(latest.value()->config().version, "10");
  auto v2 = manager.GetSession("ref", "2");
  ASSERT_TRUE(v2.ok());
  EXPECT_EQ(v2.value()->metadata().version, "2");
  EXPECT_EQ(manager.GetSession("ref", "3").status().code(), StatusCode::kNotFound);
  EXPECT_EQ(manager.GetSession("nope").status().code(), StatusCode::kNotFound);
}

TEST(SessionManagerTest, UnloadKeepsInFlightSessionAlive) {
  SessionManager manager(nullptr);
  ASSERT_TRUE(manager.LoadModel(Reference("1")).ok());
  auto held = manager.GetSession("ref", "1");
  ASSERT_TRUE(held.ok());
  ASSERT_TRUE(manager.UnloadModel("ref", "1").ok());
  EXPECT_EQ(manager.GetSession("ref", "1").status().code(), StatusCode::kNotFound);
  // The held pointer remains valid; the session reports it is no longer serving.
  EXPECT_EQ(held.value()->state(), ModelState::kUnloading);
  EXPECT_EQ(manager.UnloadModel("ref", "1").code(), StatusCode::kNotFound);
}

TEST(SessionManagerTest, FailedLoadIsReported) {
  SessionManager manager(nullptr);
  ModelConfig bad = Reference("1");
  bad.backend = "onnxruntime";
  bad.path = "/nonexistent/model.onnx";
  EXPECT_FALSE(manager.LoadModel(bad).ok());
  bool found = false;
  for (const auto& s : manager.ListStatus()) {
    if (s.model_name == "ref" && s.state == ModelState::kFailed) found = true;
  }
  EXPECT_TRUE(found);
}

TEST(SessionManagerTest, VersionOrdering) {
  EXPECT_TRUE(SessionManager::VersionLess("2", "10"));
  EXPECT_FALSE(SessionManager::VersionLess("10", "2"));
  EXPECT_TRUE(SessionManager::VersionLess("a", "b"));
}

#ifdef MIE_WITH_ONNXRUNTIME
class OnnxRuntimeBackendTest : public ::testing::Test {
 protected:
  ModelConfig Config(const std::string& file) {
    ModelConfig c;
    c.name = "mlp";
    c.backend = "onnxruntime";
    c.path = std::string(MIE_TEST_MODELS_DIR) + "/" + file;
    c.batching.max_batch_size = 32;
    return c;
  }
};

TEST_F(OnnxRuntimeBackendTest, LoadsModelMetadata) {
  auto backend = CreateBackend("onnxruntime");
  ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend.value()->Load(Config("mlp_v1.onnx")).ok());
  const auto& md = backend.value()->metadata();
  ASSERT_EQ(md.inputs.size(), 1u);
  EXPECT_EQ(md.inputs[0].name, "input");
  EXPECT_EQ(md.inputs[0].dtype, DataType::kFloat32);
  EXPECT_EQ(md.inputs[0].shape, (Shape{-1, 16}));
  ASSERT_EQ(md.outputs.size(), 2u);
  EXPECT_GT(md.max_batch_size, 0);
  EXPECT_NE(md.platform.find("onnxruntime"), std::string::npos);
}

TEST_F(OnnxRuntimeBackendTest, InferenceProducesProbabilities) {
  auto backend = CreateBackend("onnxruntime");
  ASSERT_TRUE(backend.value()->Load(Config("mlp_v1.onnx")).ok());
  std::vector<float> x(5 * 16);
  for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i));
  TensorMap in{{"input", Tensor::FromVector<float>({5, 16}, x)}};
  TensorMap out;
  ASSERT_TRUE(backend.value()->Infer(in, &out).ok());
  const Tensor& p = out.at("probabilities");
  ASSERT_EQ(p.shape(), (Shape{5, 4}));
  for (int r = 0; r < 5; ++r) {
    float sum = 0;
    for (int c = 0; c < 4; ++c) sum += p.data_as<float>()[r * 4 + c];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }
}

TEST_F(OnnxRuntimeBackendTest, BatchedEqualsUnbatchedThroughScheduler) {
  ModelConfig cfg = Config("mlp_v1.onnx");
  cfg.batching.max_queue_delay_us = 3000;
  auto session = ModelSession::Create(cfg, nullptr);
  ASSERT_TRUE(session.ok()) << session.status().ToString();
  auto direct = CreateBackend("onnxruntime");
  ASSERT_TRUE(direct.value()->Load(cfg).ok());

  std::vector<std::future<InferenceResult>> fs;
  std::vector<Tensor> inputs;
  for (int i = 0; i < 20; ++i) {
    std::vector<float> x(16);
    for (int j = 0; j < 16; ++j) x[static_cast<size_t>(j)] = static_cast<float>(i * 16 + j) / 100.0f;
    inputs.push_back(Tensor::FromVector<float>({1, 16}, x));
    fs.push_back(session.value()->InferAsync({{"input", inputs.back()}}));
  }
  bool batched = false;
  for (int i = 0; i < 20; ++i) {
    auto r = fs[static_cast<size_t>(i)].get();
    ASSERT_TRUE(r.status.ok()) << r.status.ToString();
    batched = batched || r.timing.batch_size > 1;
    TensorMap expected;
    ASSERT_TRUE(direct.value()->Infer({{"input", inputs[static_cast<size_t>(i)]}}, &expected).ok());
    const auto got = r.outputs.at("logits").ToVector<float>();
    const auto want = expected.at("logits").ToVector<float>();
    ASSERT_EQ(got.size(), want.size());
    for (size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], want[k], 1e-5f);
  }
  EXPECT_TRUE(batched);
}

TEST_F(OnnxRuntimeBackendTest, MissingFileFails) {
  auto backend = CreateBackend("onnxruntime");
  EXPECT_FALSE(backend.value()->Load(Config("missing.onnx")).ok());
}
#endif  // MIE_WITH_ONNXRUNTIME

}  // namespace
}  // namespace mie
