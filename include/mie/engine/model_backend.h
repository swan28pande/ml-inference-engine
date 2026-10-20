// Abstract execution backend. One ModelBackend instance executes one batch at
// a time; ModelSession creates `instance_count` instances to run batches in
// parallel, which is how thread-safe access is provided for runtimes whose
// execution contexts are not re-entrant (e.g. TensorRT IExecutionContext).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mie/common/status.h"
#include "mie/common/tensor.h"
#include "mie/engine/model_config.h"

namespace mie {

class ModelBackend {
 public:
  virtual ~ModelBackend() = default;

  // Loads the model artifact and fills metadata(). Called once.
  virtual Status Load(const ModelConfig& config) = 0;

  // Runs a (possibly batched) forward pass. All inputs share dimension 0.
  virtual Status Infer(const TensorMap& inputs, TensorMap* outputs) = 0;

  virtual const ModelMetadata& metadata() const = 0;

  // Runs `iterations` forward passes on zero-filled inputs of `batch_size`
  // samples so that lazy initialisation (kernel autotuning, allocator growth)
  // happens before serving traffic.
  virtual Status Warmup(int batch_size, int iterations);
};

// Creates an unloaded backend by name. Names of backends compiled into this
// binary are returned by AvailableBackends().
StatusOr<std::unique_ptr<ModelBackend>> CreateBackend(const std::string& name);
std::vector<std::string> AvailableBackends();

// Builds zero-filled inputs matching `specs` with dynamic dims replaced by
// `batch_size` (dim 0) or 1 (others).
TensorMap MakeZeroInputs(const std::vector<TensorSpec>& specs, int batch_size);

}  // namespace mie
