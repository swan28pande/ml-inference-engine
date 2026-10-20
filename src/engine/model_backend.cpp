#include "mie/engine/model_backend.h"

#include <algorithm>
#include <cctype>

#include "mie/engine/backends/reference_backend.h"

#ifdef MIE_WITH_ONNXRUNTIME
#include "mie/engine/backends/onnxruntime_backend.h"
#endif
#if defined(MIE_WITH_TENSORRT) && defined(MIE_WITH_CUDA)
#include "mie/engine/backends/tensorrt_backend.h"
#endif
#ifdef MIE_WITH_TORCH
#include "mie/engine/backends/torchscript_backend.h"
#endif

namespace mie {

int64_t ModelConfig::GetIntParameter(const std::string& key, int64_t fallback) const {
  auto it = parameters.find(key);
  if (it == parameters.end() || it->second.empty()) return fallback;
  try {
    return std::stoll(it->second);
  } catch (...) {
    return fallback;
  }
}

bool ModelConfig::GetBoolParameter(const std::string& key, bool fallback) const {
  auto it = parameters.find(key);
  if (it == parameters.end()) return fallback;
  std::string v = it->second;
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  return fallback;
}

TensorMap MakeZeroInputs(const std::vector<TensorSpec>& specs, int batch_size) {
  TensorMap inputs;
  for (const auto& spec : specs) {
    Shape shape = spec.shape;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] < 0) shape[i] = (i == 0) ? batch_size : 1;
    }
    inputs.emplace(spec.name, Tensor(spec.dtype, std::move(shape)));
  }
  return inputs;
}

Status ModelBackend::Warmup(int batch_size, int iterations) {
  const ModelMetadata& md = metadata();
  if (md.inputs.empty()) return Status::Ok();
  if (md.max_batch_size == 0) batch_size = 1;
  TensorMap inputs = MakeZeroInputs(md.inputs, batch_size);
  TensorMap outputs;
  for (int i = 0; i < iterations; ++i) {
    outputs.clear();
    MIE_RETURN_IF_ERROR(Infer(inputs, &outputs));
  }
  return Status::Ok();
}

StatusOr<std::unique_ptr<ModelBackend>> CreateBackend(const std::string& name) {
  if (name == "reference") return std::unique_ptr<ModelBackend>(std::make_unique<ReferenceBackend>());
#ifdef MIE_WITH_ONNXRUNTIME
  if (name == "onnxruntime" || name == "onnx") {
    return std::unique_ptr<ModelBackend>(std::make_unique<OnnxRuntimeBackend>());
  }
#endif
#if defined(MIE_WITH_TENSORRT) && defined(MIE_WITH_CUDA)
  if (name == "tensorrt") return std::unique_ptr<ModelBackend>(std::make_unique<TensorRTBackend>());
#endif
#ifdef MIE_WITH_TORCH
  if (name == "torchscript" || name == "pytorch") {
    return std::unique_ptr<ModelBackend>(std::make_unique<TorchScriptBackend>());
  }
#endif
  std::string available;
  for (const auto& b : AvailableBackends()) available += (available.empty() ? "" : ", ") + b;
  return Unimplemented("backend '" + name + "' is not available in this build (available: " + available + ")");
}

std::vector<std::string> AvailableBackends() {
  std::vector<std::string> out{"reference"};
#ifdef MIE_WITH_ONNXRUNTIME
  out.push_back("onnxruntime");
#endif
#if defined(MIE_WITH_TENSORRT) && defined(MIE_WITH_CUDA)
  out.push_back("tensorrt");
#endif
#ifdef MIE_WITH_TORCH
  out.push_back("torchscript");
#endif
  return out;
}

}  // namespace mie
