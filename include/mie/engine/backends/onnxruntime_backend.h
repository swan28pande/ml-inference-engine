// ONNX Runtime backend (CPU, CUDA or TensorRT execution providers).
//
// Parameters:
//   execution_provider  cpu | cuda | tensorrt   (default: cuda if use_gpu else cpu)
//   intra_op_threads    threads per op on CPU   (default 1: parallelism comes from batching + instances)
//   inter_op_threads    (default 1)
//   graph_optimization  disable | basic | extended | all (default all)
//   trt_fp16            enable FP16 kernels in the TensorRT EP (default 1)
//   trt_engine_cache    directory for cached TensorRT engines (default: next to the model)
//   trt_max_workspace   bytes (default 1 GiB)
//
// Ort::Session::Run is thread-safe, but each ModelBackend instance still
// executes one batch at a time; use instance_count for parallel execution.
#pragma once

#ifdef MIE_WITH_ONNXRUNTIME

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "mie/engine/model_backend.h"

namespace mie {

class OnnxRuntimeBackend final : public ModelBackend {
 public:
  OnnxRuntimeBackend();
  ~OnnxRuntimeBackend() override;

  Status Load(const ModelConfig& config) override;
  Status Infer(const TensorMap& inputs, TensorMap* outputs) override;
  const ModelMetadata& metadata() const override { return metadata_; }

 private:
  ModelMetadata metadata_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> cpu_memory_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
};

}  // namespace mie

#endif  // MIE_WITH_ONNXRUNTIME
