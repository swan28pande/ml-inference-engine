// Native TensorRT backend (TensorRT 10.x API).
//
// Accepts either a serialized engine (*.engine / *.plan) or an ONNX model,
// which is compiled into an engine on first load and cached on disk. Device
// I/O buffers come from the per-GPU MemoryPool, so steady-state inference
// performs no cudaMalloc/cudaFree. FP16 engines keep a float32 client API:
// conversion happens on the GPU with the kernels in cuda_kernels.cu.
//
// Parameters:
//   fp16              build with FP16 kernels (default 1)
//   workspace_bytes   builder workspace limit (default 1 GiB)
//   min_batch / opt_batch / max_batch   optimisation profile for dim 0
//                     (defaults 1 / batching.max_batch_size / batching.max_batch_size)
//   engine_cache      path of the cached engine (default: <model>.<fp16|fp32>.engine)
//   rebuild_engine    ignore an existing cache (default 0)
//
// IExecutionContext is not thread-safe, so each instance serialises its own
// Infer() calls; configure instance_count > 1 for concurrent execution.
#pragma once

#if defined(MIE_WITH_TENSORRT) && defined(MIE_WITH_CUDA)

#include <NvInfer.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mie/engine/model_backend.h"
#include "mie/memory/cuda_utils.h"
#include "mie/memory/memory_pool.h"

namespace mie {

class TensorRTBackend final : public ModelBackend {
 public:
  TensorRTBackend();
  ~TensorRTBackend() override;

  Status Load(const ModelConfig& config) override;
  Status Infer(const TensorMap& inputs, TensorMap* outputs) override;
  const ModelMetadata& metadata() const override { return metadata_; }

 private:
  struct IoBinding {
    std::string name;
    bool is_input = false;
    nvinfer1::DataType engine_dtype = nvinfer1::DataType::kFLOAT;
    DataType client_dtype = DataType::kFloat32;  // what clients send/receive
  };

  Status BuildEngineFromOnnx(const ModelConfig& config, std::vector<char>* plan);
  Status InitBindings(const ModelConfig& config);

  ModelMetadata metadata_;
  int device_id_ = 0;
  MemoryPool* device_pool_ = nullptr;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::unique_ptr<CudaStream> stream_;
  std::vector<IoBinding> bindings_;
  std::mutex mu_;
};

}  // namespace mie

#endif  // MIE_WITH_TENSORRT && MIE_WITH_CUDA
