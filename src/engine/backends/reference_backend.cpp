#include "mie/engine/backends/reference_backend.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace mie {
namespace {

// Small deterministic PRNG so weights are identical across platforms.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}
  uint64_t Next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  // Uniform in [-1, 1).
  float NextFloat() { return static_cast<float>(Next() >> 40) / static_cast<float>(1ULL << 23) - 1.0f; }

 private:
  uint64_t state_;
};

}  // namespace

Status ReferenceBackend::Load(const ModelConfig& config) {
  input_dim_ = config.GetIntParameter("input_dim", 16);
  output_dim_ = config.GetIntParameter("output_dim", 4);
  fixed_latency_us_ = config.GetIntParameter("fixed_latency_us", 0);
  per_sample_latency_us_ = config.GetIntParameter("per_sample_latency_us", 0);
  const int64_t max_batch = config.GetIntParameter("max_batch_size", 1024);
  if (input_dim_ <= 0 || output_dim_ <= 0) return InvalidArgument("input_dim and output_dim must be positive");

  SplitMix64 rng(static_cast<uint64_t>(config.GetIntParameter("seed", 42)));
  weights_.resize(static_cast<size_t>(input_dim_ * output_dim_));
  bias_.resize(static_cast<size_t>(output_dim_));
  const float scale = 1.0f / static_cast<float>(input_dim_);
  for (auto& w : weights_) w = rng.NextFloat() * scale;
  for (auto& b : bias_) b = rng.NextFloat() * 0.1f;

  metadata_ = ModelMetadata{};
  metadata_.platform = "reference-cpu";
  metadata_.max_batch_size = static_cast<int>(max_batch);
  metadata_.inputs = {{"input", DataType::kFloat32, {-1, input_dim_}}};
  metadata_.outputs = {{"output", DataType::kFloat32, {-1, output_dim_}}};
  metadata_.properties["input_dim"] = std::to_string(input_dim_);
  metadata_.properties["output_dim"] = std::to_string(output_dim_);
  return Status::Ok();
}

std::vector<float> ReferenceBackend::Forward(const float* x) const {
  std::vector<float> y(static_cast<size_t>(output_dim_));
  for (int64_t o = 0; o < output_dim_; ++o) {
    const float* w = weights_.data() + o * input_dim_;
    float acc = bias_[static_cast<size_t>(o)];
    for (int64_t i = 0; i < input_dim_; ++i) acc += w[i] * x[i];
    y[static_cast<size_t>(o)] = std::max(acc, 0.0f);
  }
  return y;
}

Status ReferenceBackend::Infer(const TensorMap& inputs, TensorMap* outputs) {
  auto it = inputs.find("input");
  if (it == inputs.end()) return InvalidArgument("reference backend expects an input named 'input'");
  const Tensor& x = it->second;
  if (x.dtype() != DataType::kFloat32) return InvalidArgument("input must be float32");
  if (x.shape().size() != 2 || x.shape()[1] != input_dim_) {
    return InvalidArgument("input must have shape [N, " + std::to_string(input_dim_) + "], got " +
                           ShapeToString(x.shape()));
  }
  const int64_t n = x.shape()[0];
  const auto start = std::chrono::steady_clock::now();

  Tensor y(DataType::kFloat32, {n, output_dim_});
  const float* src = x.data_as<float>();
  float* dst = y.data_as<float>();
  for (int64_t s = 0; s < n; ++s) {
    const auto row = Forward(src + s * input_dim_);
    std::copy(row.begin(), row.end(), dst + s * output_dim_);
  }

  // Simulated accelerator cost: sleep for whatever is left of the budget.
  const int64_t budget_us = fixed_latency_us_ + per_sample_latency_us_ * n;
  if (budget_us > 0) {
    std::this_thread::sleep_until(start + std::chrono::microseconds(budget_us));
  }
  (*outputs)["output"] = std::move(y);
  return Status::Ok();
}

}  // namespace mie
