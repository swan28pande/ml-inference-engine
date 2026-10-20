// Dependency-free CPU backend: a deterministic dense layer y = relu(Wx + b).
//
// Used by unit tests, CI and the benchmark harness. It can also simulate an
// accelerator's cost model via two parameters, which makes the effect of
// dynamic batching measurable without a GPU:
//   fixed_latency_us      - per-batch overhead (kernel launch, H2D/D2H setup)
//   per_sample_latency_us - incremental cost per sample in the batch
//
// Parameters: input_dim (16), output_dim (4), seed (42), max_batch_size (1024),
//             fixed_latency_us (0), per_sample_latency_us (0)
#pragma once

#include <cstdint>
#include <vector>

#include "mie/engine/model_backend.h"

namespace mie {

class ReferenceBackend final : public ModelBackend {
 public:
  Status Load(const ModelConfig& config) override;
  Status Infer(const TensorMap& inputs, TensorMap* outputs) override;
  const ModelMetadata& metadata() const override { return metadata_; }

  // Unbatched reference computation for a single sample (used by tests).
  std::vector<float> Forward(const float* x) const;

 private:
  ModelMetadata metadata_;
  int64_t input_dim_ = 16;
  int64_t output_dim_ = 4;
  int64_t fixed_latency_us_ = 0;
  int64_t per_sample_latency_us_ = 0;
  std::vector<float> weights_;  // [output_dim, input_dim]
  std::vector<float> bias_;     // [output_dim]
};

}  // namespace mie
