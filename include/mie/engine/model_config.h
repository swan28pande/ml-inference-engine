// Engine-level model configuration and metadata (independent of protobuf).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mie/common/tensor.h"

namespace mie {

struct TensorSpec {
  std::string name;
  DataType dtype = DataType::kFloat32;
  Shape shape;  // -1 marks a dynamic dimension; dim 0 is the batch dimension
};

struct BatchingConfig {
  bool enabled = true;
  // Max number of samples (sum of request batch dims) in one executed batch.
  int max_batch_size = 32;
  // Upper bound on how long the oldest queued request waits for the batch to fill.
  int64_t max_queue_delay_us = 1000;
  // If set, batches dispatched on timeout are trimmed to the largest preferred
  // size that fits (e.g. TensorRT optimisation profile sizes).
  std::vector<int> preferred_batch_sizes;
  // Pending requests allowed per model version before rejecting with RESOURCE_EXHAUSTED.
  size_t max_queue_size = 4096;
};

struct ModelConfig {
  std::string name;
  std::string version = "1";
  std::string backend;  // onnxruntime | tensorrt | torchscript | reference
  std::string path;     // model artifact on disk
  int instance_count = 1;
  BatchingConfig batching;
  bool use_gpu = false;
  int device_id = 0;
  int warmup_iterations = 0;
  std::map<std::string, std::string> parameters;  // backend-specific options
  std::vector<TensorSpec> inputs;                  // optional; required by some backends
  std::vector<TensorSpec> outputs;

  std::string GetParameter(const std::string& key, const std::string& fallback = "") const {
    auto it = parameters.find(key);
    return it == parameters.end() ? fallback : it->second;
  }
  int64_t GetIntParameter(const std::string& key, int64_t fallback) const;
  bool GetBoolParameter(const std::string& key, bool fallback) const;
};

struct ModelMetadata {
  std::string name;
  std::string version;
  std::string backend;
  std::string platform;  // e.g. "onnxruntime-1.22 (CUDAExecutionProvider)"
  // 0 means the model has no dynamic batch dimension (requests run one at a time).
  int max_batch_size = 0;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::map<std::string, std::string> properties;
};

}  // namespace mie
