// End-to-end tests: an in-process InferenceServer on an ephemeral port,
// exercised through the generated gRPC client stub.
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <cstring>
#include <future>
#include <thread>

#include "inference.grpc.pb.h"
#include "mie/server/grpc_server.h"

namespace mie {
namespace {

ModelConfig ReferenceModel(const std::string& version, int fixed_latency_us = 0) {
  ModelConfig c;
  c.name = "ref";
  c.version = version;
  c.backend = "reference";
  c.batching.max_batch_size = 16;
  c.batching.max_queue_delay_us = 500;
  c.parameters["input_dim"] = "4";
  c.parameters["output_dim"] = "2";
  c.parameters["seed"] = version;
  c.parameters["fixed_latency_us"] = std::to_string(fixed_latency_us);
  return c;
}

class GrpcServerTest : public ::testing::Test {
 protected:
  void StartServer(std::vector<ModelConfig> models, std::map<std::string, TrafficWeights> traffic = {}) {
    ServerOptions opts;
    opts.listen_address = "127.0.0.1:0";
    opts.drain_timeout = std::chrono::milliseconds(5000);
    opts.models = std::move(models);
    opts.traffic = std::move(traffic);
    server_ = std::make_unique<InferenceServer>(std::move(opts));
    ASSERT_TRUE(server_->Start().ok());
    ASSERT_GT(server_->port(), 0);
    channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(server_->port()), grpc::InsecureChannelCredentials());
    stub_ = v1::InferenceService::NewStub(channel_);
  }

  void TearDown() override {
    if (server_) server_->Shutdown();
  }

  static v1::PredictRequest MakePredict(int batch, const std::string& version = "", const std::string& key = "") {
    v1::PredictRequest req;
    req.set_model_name("ref");
    req.set_model_version(version);
    req.set_routing_key(key);
    auto* t = req.add_inputs();
    t->set_name("input");
    t->set_dtype(v1::DT_FLOAT32);
    t->add_shape(batch);
    t->add_shape(4);
    std::vector<float> data(static_cast<size_t>(batch) * 4, 0.5f);
    t->set_raw_data(data.data(), data.size() * sizeof(float));
    return req;
  }

  grpc::Status Predict(const v1::PredictRequest& req, v1::PredictResponse* resp) {
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    return stub_->Predict(&ctx, req, resp);
  }

  std::unique_ptr<InferenceServer> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<v1::InferenceService::Stub> stub_;
};

TEST_F(GrpcServerTest, HealthReportsServingModels) {
  StartServer({ReferenceModel("1"), ReferenceModel("2")});
  grpc::ClientContext ctx;
  v1::HealthResponse resp;
  ASSERT_TRUE(stub_->Health(&ctx, v1::HealthRequest(), &resp).ok());
  EXPECT_EQ(resp.status(), v1::SERVING);
  ASSERT_EQ(resp.models_size(), 2);
  for (const auto& m : resp.models()) EXPECT_EQ(m.state(), v1::READY);
  EXPECT_FALSE(resp.server_version().empty());
}

TEST_F(GrpcServerTest, PredictReturnsOutputsAndTiming) {
  StartServer({ReferenceModel("1")});
  v1::PredictResponse resp;
  auto status = Predict(MakePredict(3), &resp);
  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_EQ(resp.model_version(), "1");
  ASSERT_EQ(resp.outputs_size(), 1);
  const auto& out = resp.outputs(0);
  EXPECT_EQ(out.name(), "output");
  ASSERT_EQ(out.shape_size(), 2);
  EXPECT_EQ(out.shape(0), 3);
  EXPECT_EQ(out.shape(1), 2);
  EXPECT_EQ(out.raw_data().size(), 3u * 2u * sizeof(float));
  EXPECT_GE(resp.timing().batch_size(), 3);
  EXPECT_GT(resp.timing().total_us(), 0);
}

TEST_F(GrpcServerTest, PredictErrors) {
  StartServer({ReferenceModel("1")});
  v1::PredictResponse resp;

  auto unknown = MakePredict(1);
  unknown.set_model_name("nope");
  EXPECT_EQ(Predict(unknown, &resp).error_code(), grpc::StatusCode::NOT_FOUND);

  auto bad_version = MakePredict(1, "9");
  EXPECT_EQ(Predict(bad_version, &resp).error_code(), grpc::StatusCode::NOT_FOUND);

  auto truncated = MakePredict(2);
  truncated.mutable_inputs(0)->mutable_raw_data()->resize(5);
  EXPECT_EQ(Predict(truncated, &resp).error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  auto too_big = MakePredict(64);  // max_batch_size is 16
  EXPECT_EQ(Predict(too_big, &resp).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(GrpcServerTest, MetadataAndListModels) {
  StartServer({ReferenceModel("1"), ReferenceModel("2")}, {{"ref", {{"1", 70}, {"2", 30}}}});
  {
    grpc::ClientContext ctx;
    v1::ModelMetadataRequest req;
    req.set_model_name("ref");
    v1::ModelMetadataResponse resp;
    ASSERT_TRUE(stub_->GetModelMetadata(&ctx, req, &resp).ok());
    EXPECT_EQ(resp.version(), "2");  // latest by default
    EXPECT_EQ(resp.backend(), "reference");
    EXPECT_EQ(resp.max_batch_size(), 16);
    ASSERT_EQ(resp.inputs_size(), 1);
    EXPECT_EQ(resp.inputs(0).shape(1), 4);
  }
  {
    grpc::ClientContext ctx;
    v1::ListModelsResponse resp;
    ASSERT_TRUE(stub_->ListModels(&ctx, v1::ListModelsRequest(), &resp).ok());
    ASSERT_EQ(resp.models_size(), 1);
    EXPECT_EQ(resp.models(0).versions_size(), 2);
    EXPECT_EQ(resp.models(0).traffic_weights().at("1"), 70u);
  }
}

TEST_F(GrpcServerTest, ABRoutingAndRuntimeSplitUpdates) {
  StartServer({ReferenceModel("1"), ReferenceModel("2")}, {{"ref", {{"1", 50}, {"2", 50}}}});
  std::map<std::string, int> counts;
  for (int i = 0; i < 200; ++i) {
    v1::PredictResponse resp;
    ASSERT_TRUE(Predict(MakePredict(1), &resp).ok());
    counts[resp.model_version()]++;
  }
  EXPECT_GT(counts["1"], 40);
  EXPECT_GT(counts["2"], 40);

  // Sticky routing: the same key always hits the same version.
  std::string first;
  for (int i = 0; i < 20; ++i) {
    v1::PredictResponse resp;
    ASSERT_TRUE(Predict(MakePredict(1, "", "user-42"), &resp).ok());
    if (first.empty()) first = resp.model_version();
    EXPECT_EQ(resp.model_version(), first);
  }

  // Shift 100% of traffic to v2 at runtime.
  grpc::ClientContext ctx;
  v1::SetTrafficSplitRequest split;
  split.set_model_name("ref");
  (*split.mutable_weights())["2"] = 100;
  v1::SetTrafficSplitResponse split_resp;
  ASSERT_TRUE(stub_->SetTrafficSplit(&ctx, split, &split_resp).ok());
  for (int i = 0; i < 50; ++i) {
    v1::PredictResponse resp;
    ASSERT_TRUE(Predict(MakePredict(1), &resp).ok());
    EXPECT_EQ(resp.model_version(), "2");
  }

  // Splits referencing unknown versions are rejected.
  grpc::ClientContext ctx2;
  (*split.mutable_weights())["99"] = 1;
  EXPECT_EQ(stub_->SetTrafficSplit(&ctx2, split, &split_resp).error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(GrpcServerTest, MetricsEndpoint) {
  StartServer({ReferenceModel("1")});
  for (int i = 0; i < 20; ++i) {
    v1::PredictResponse resp;
    ASSERT_TRUE(Predict(MakePredict(1), &resp).ok());
  }
  grpc::ClientContext ctx;
  v1::MetricsRequest req;
  req.set_prometheus_format(true);
  v1::MetricsResponse resp;
  ASSERT_TRUE(stub_->GetMetrics(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.models_size(), 1);
  EXPECT_EQ(resp.models(0).requests_ok(), 20u);
  EXPECT_GT(resp.models(0).request_latency().p99_us(), 0);
  EXPECT_NE(resp.prometheus_text().find("mie_requests_total"), std::string::npos);
}

TEST_F(GrpcServerTest, GracefulShutdownDrainsInFlightRequests) {
  StartServer({ReferenceModel("1", /*fixed_latency_us=*/200000)});  // 200 ms per batch
  constexpr int kClients = 8;
  std::vector<std::future<grpc::Status>> inflight;
  for (int i = 0; i < kClients; ++i) {
    inflight.push_back(std::async(std::launch::async, [this] {
      v1::PredictResponse resp;
      return Predict(MakePredict(1), &resp);
    }));
  }
  // Wait until the requests are admitted, then shut down while they run.
  for (int i = 0; i < 100 && server_->limiter()->in_flight() < kClients; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_EQ(server_->limiter()->in_flight(), static_cast<size_t>(kClients));
  std::thread shutdown([this] { server_->Shutdown(); });

  for (auto& f : inflight) {
    const grpc::Status s = f.get();
    EXPECT_TRUE(s.ok()) << "in-flight request was dropped: " << s.error_message();
  }
  shutdown.join();
  EXPECT_EQ(server_->limiter()->in_flight(), 0u);

  v1::PredictResponse resp;
  EXPECT_FALSE(Predict(MakePredict(1), &resp).ok());  // no longer serving
}

TEST_F(GrpcServerTest, ConcurrencyLimitRejectsExcessRequests) {
  ServerOptions opts;
  opts.listen_address = "127.0.0.1:0";
  opts.max_concurrent_requests = 2;
  opts.models = {ReferenceModel("1", /*fixed_latency_us=*/100000)};
  server_ = std::make_unique<InferenceServer>(std::move(opts));
  ASSERT_TRUE(server_->Start().ok());
  channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(server_->port()), grpc::InsecureChannelCredentials());
  stub_ = v1::InferenceService::NewStub(channel_);

  std::vector<std::future<grpc::Status>> fs;
  for (int i = 0; i < 8; ++i) {
    fs.push_back(std::async(std::launch::async, [this] {
      v1::PredictResponse resp;
      return Predict(MakePredict(1), &resp);
    }));
  }
  int exhausted = 0, ok = 0;
  for (auto& f : fs) {
    const auto s = f.get();
    exhausted += s.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED;
    ok += s.ok();
  }
  EXPECT_GE(ok, 2);
  EXPECT_GE(exhausted, 1);
}

}  // namespace
}  // namespace mie
