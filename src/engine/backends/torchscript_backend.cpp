#include "mie/engine/backends/torchscript_backend.h"

#ifdef MIE_WITH_TORCH

#include <torch/version.h>

#include <cstring>

namespace mie {
namespace {

StatusOr<torch::ScalarType> ToTorchType(DataType t) {
  switch (t) {
    case DataType::kFloat32: return torch::kFloat32;
    case DataType::kFloat16: return torch::kFloat16;
    case DataType::kInt64: return torch::kInt64;
    case DataType::kInt32: return torch::kInt32;
    case DataType::kUint8: return torch::kUInt8;
    case DataType::kBool: return torch::kBool;
    case DataType::kInvalid: break;
  }
  return InvalidArgument("unsupported dtype for torchscript");
}

DataType FromTorchType(torch::ScalarType t) {
  switch (t) {
    case torch::kFloat32: return DataType::kFloat32;
    case torch::kFloat16: return DataType::kFloat32;  // returned to clients as float32
    case torch::kInt64: return DataType::kInt64;
    case torch::kInt32: return DataType::kInt32;
    case torch::kUInt8: return DataType::kUint8;
    case torch::kBool: return DataType::kBool;
    default: return DataType::kInvalid;
  }
}

}  // namespace

Status TorchScriptBackend::Load(const ModelConfig& config) {
  if (config.path.empty()) return InvalidArgument("torchscript backend requires a model path");
  if (config.inputs.empty()) {
    return InvalidArgument("torchscript models need explicit `inputs` in the model config");
  }
  try {
    device_ = config.use_gpu ? torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(config.device_id))
                             : torch::Device(torch::kCPU);
    module_ = torch::jit::load(config.path, device_);
    module_.eval();
    fp16_ = config.GetBoolParameter("fp16", false);
    if (fp16_) module_.to(torch::kHalf);
  } catch (const c10::Error& e) {
    return Internal(std::string("torch::jit::load failed: ") + e.what_without_backtrace());
  }

  metadata_ = ModelMetadata{};
  metadata_.inputs = config.inputs;
  metadata_.outputs = config.outputs;
  if (metadata_.outputs.empty()) metadata_.outputs.push_back({"output0", DataType::kFloat32, {-1}});
  bool dynamic = true;
  for (const auto& in : metadata_.inputs) dynamic = dynamic && !in.shape.empty() && in.shape[0] < 0;
  metadata_.max_batch_size = dynamic ? static_cast<int>(config.GetIntParameter("max_batch_size", 1024)) : 0;
  metadata_.platform = std::string("libtorch-") + TORCH_VERSION + (device_.is_cuda() ? " (cuda)" : " (cpu)");
  return Status::Ok();
}

Status TorchScriptBackend::Infer(const TensorMap& inputs, TensorMap* outputs) {
  std::lock_guard<std::mutex> lock(mu_);
  try {
    c10::InferenceMode guard;
    std::vector<torch::jit::IValue> args;
    args.reserve(metadata_.inputs.size());
    for (const auto& spec : metadata_.inputs) {
      auto it = inputs.find(spec.name);
      if (it == inputs.end()) return InvalidArgument("missing model input '" + spec.name + "'");
      const Tensor& t = it->second;
      MIE_ASSIGN_OR_RETURN(torch::ScalarType st, ToTorchType(t.dtype()));
      std::vector<int64_t> sizes(t.shape().begin(), t.shape().end());
      torch::Tensor x = torch::from_blob(const_cast<void*>(t.data()), sizes, torch::TensorOptions().dtype(st));
      x = x.to(device_, /*non_blocking=*/false, /*copy=*/true);
      if (fp16_ && x.is_floating_point()) x = x.to(torch::kHalf);
      args.emplace_back(std::move(x));
    }

    torch::jit::IValue result = module_.forward(args);
    std::vector<torch::Tensor> tensors;
    if (result.isTensor()) {
      tensors.push_back(result.toTensor());
    } else if (result.isTuple()) {
      for (const auto& e : result.toTupleRef().elements()) tensors.push_back(e.toTensor());
    } else if (result.isTensorList()) {
      for (const auto& e : result.toTensorVector()) tensors.push_back(e);
    } else {
      return Unimplemented("torchscript model must return a tensor, tuple or list of tensors");
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
      torch::Tensor y = tensors[i].to(torch::kCPU);
      if (y.scalar_type() == torch::kHalf) y = y.to(torch::kFloat32);
      y = y.contiguous();
      const DataType dtype = FromTorchType(y.scalar_type());
      if (dtype == DataType::kInvalid) return Unimplemented("unsupported output dtype");
      Shape shape(y.sizes().begin(), y.sizes().end());
      Tensor out(dtype, shape);
      if (out.byte_size()) std::memcpy(out.data(), y.data_ptr(), out.byte_size());
      const std::string name = i < metadata_.outputs.size() ? metadata_.outputs[i].name : "output" + std::to_string(i);
      (*outputs)[name] = std::move(out);
    }
  } catch (const c10::Error& e) {
    return Internal(std::string("torchscript: ") + e.what_without_backtrace());
  }
  return Status::Ok();
}

}  // namespace mie

#endif  // MIE_WITH_TORCH
