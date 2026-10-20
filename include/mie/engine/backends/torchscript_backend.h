// TorchScript (LibTorch) backend.
//
// TorchScript modules carry no I/O signature, so the model config must list
// `inputs` (positional order of forward() arguments) and may list `outputs`
// (names for the returned tensor / tuple elements; default output0..N).
//
// Parameters: fp16 (default 0) - cast the module and float inputs to half.
#pragma once

#ifdef MIE_WITH_TORCH

#include <torch/script.h>

#include <mutex>

#include "mie/engine/model_backend.h"

namespace mie {

class TorchScriptBackend final : public ModelBackend {
 public:
  Status Load(const ModelConfig& config) override;
  Status Infer(const TensorMap& inputs, TensorMap* outputs) override;
  const ModelMetadata& metadata() const override { return metadata_; }

 private:
  ModelMetadata metadata_;
  torch::jit::script::Module module_;
  torch::Device device_{torch::kCPU};
  bool fp16_ = false;
  std::mutex mu_;
};

}  // namespace mie

#endif  // MIE_WITH_TORCH
