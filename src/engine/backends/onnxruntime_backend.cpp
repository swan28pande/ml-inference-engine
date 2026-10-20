#include "mie/engine/backends/onnxruntime_backend.h"

#ifdef MIE_WITH_ONNXRUNTIME

#include <algorithm>
#include <cstring>

#include "mie/common/logging.h"

namespace mie {
namespace {

Ort::Env& GetEnv() {
  // One environment per process; ORT shares thread pools and logging through it.
  // A function-local static (not a leaked pointer) so it is released before
  // ONNX Runtime's own globals are torn down at exit.
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mie");
  return env;
}

DataType FromOrtType(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DataType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DataType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DataType::kUint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return DataType::kBool;
    default: return DataType::kInvalid;
  }
}

ONNXTensorElementDataType ToOrtType(DataType t) {
  switch (t) {
    case DataType::kFloat32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::kFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::kInt64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::kInt32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::kUint8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::kBool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case DataType::kInvalid: break;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

GraphOptimizationLevel ParseOptLevel(const std::string& s) {
  if (s == "disable") return GraphOptimizationLevel::ORT_DISABLE_ALL;
  if (s == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (s == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  return GraphOptimizationLevel::ORT_ENABLE_ALL;
}

std::string DirName(const std::string& path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? "." : path.substr(0, pos);
}

void AppendTensorRTProvider(Ort::SessionOptions& opts, const ModelConfig& config) {
  const OrtApi& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* trt = nullptr;
  Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
  const std::string device = std::to_string(config.device_id);
  const std::string fp16 = config.GetBoolParameter("trt_fp16", true) ? "1" : "0";
  const std::string cache_dir = config.GetParameter("trt_engine_cache", DirName(config.path));
  const std::string workspace = config.GetParameter("trt_max_workspace", std::to_string(size_t{1} << 30));
  const char* keys[] = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable", "trt_engine_cache_path",
                        "trt_max_workspace_size"};
  const char* values[] = {device.c_str(), fp16.c_str(), "1", cache_dir.c_str(), workspace.c_str()};
  Ort::Status update(api.UpdateTensorRTProviderOptions(trt, keys, values, sizeof(keys) / sizeof(keys[0])));
  if (!update.IsOK()) {
    api.ReleaseTensorRTProviderOptions(trt);
    throw Ort::Exception(update.GetErrorMessage(), update.GetErrorCode());
  }
  opts.AppendExecutionProvider_TensorRT_V2(*trt);
  api.ReleaseTensorRTProviderOptions(trt);
}

}  // namespace

OnnxRuntimeBackend::OnnxRuntimeBackend() = default;
OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

Status OnnxRuntimeBackend::Load(const ModelConfig& config) {
  if (config.path.empty()) return InvalidArgument("onnxruntime backend requires a model path");
  const std::string provider = config.GetParameter("execution_provider", config.use_gpu ? "cuda" : "cpu");

  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(static_cast<int>(config.GetIntParameter("intra_op_threads", 1)));
    opts.SetInterOpNumThreads(static_cast<int>(config.GetIntParameter("inter_op_threads", 1)));
    opts.SetGraphOptimizationLevel(ParseOptLevel(config.GetParameter("graph_optimization", "all")));
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (provider == "tensorrt") AppendTensorRTProvider(opts, config);
    if (provider == "tensorrt" || provider == "cuda") {
      // CUDA EP also serves as the fallback for nodes TensorRT cannot handle.
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = config.device_id;
      cuda.arena_extend_strategy = 1;  // kSameAsRequested: keep the arena tight
      cuda.do_copy_in_default_stream = 1;
      opts.AppendExecutionProvider_CUDA(cuda);
    } else if (provider != "cpu") {
      return InvalidArgument("unknown execution_provider '" + provider + "'");
    }

    session_ = std::make_unique<Ort::Session>(GetEnv(), config.path.c_str(), opts);
    cpu_memory_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    Ort::AllocatorWithDefaultOptions allocator;
    metadata_ = ModelMetadata{};
    bool dynamic_batch = true;
    auto read_spec = [&](Ort::TypeInfo type_info, std::string name, TensorSpec* spec) -> Status {
      if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        return Unimplemented("non-tensor model input/output '" + name + "' is not supported");
      }
      auto info = type_info.GetTensorTypeAndShapeInfo();
      spec->name = std::move(name);
      spec->dtype = FromOrtType(info.GetElementType());
      spec->shape = info.GetShape();
      if (spec->dtype == DataType::kInvalid) return Unimplemented("unsupported element type for '" + spec->name + "'");
      return Status::Ok();
    };

    input_names_.clear();
    output_names_.clear();
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
      TensorSpec spec;
      MIE_RETURN_IF_ERROR(read_spec(session_->GetInputTypeInfo(i),
                                    session_->GetInputNameAllocated(i, allocator).get(), &spec));
      if (spec.shape.empty() || spec.shape[0] >= 0) dynamic_batch = false;
      input_names_.push_back(spec.name);
      metadata_.inputs.push_back(std::move(spec));
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
      TensorSpec spec;
      MIE_RETURN_IF_ERROR(read_spec(session_->GetOutputTypeInfo(i),
                                    session_->GetOutputNameAllocated(i, allocator).get(), &spec));
      output_names_.push_back(spec.name);
      metadata_.outputs.push_back(std::move(spec));
    }

    input_name_ptrs_.clear();
    output_name_ptrs_.clear();
    for (const auto& n : input_names_) input_name_ptrs_.push_back(n.c_str());
    for (const auto& n : output_names_) output_name_ptrs_.push_back(n.c_str());

    metadata_.max_batch_size = dynamic_batch ? static_cast<int>(config.GetIntParameter("max_batch_size", 1024)) : 0;
    metadata_.platform = std::string("onnxruntime-") + OrtGetApiBase()->GetVersionString() + " (" + provider + ")";
    metadata_.properties["execution_provider"] = provider;

    Ort::ModelMetadata md = session_->GetModelMetadata();
    metadata_.properties["producer"] = md.GetProducerNameAllocated(allocator).get();
    metadata_.properties["graph_name"] = md.GetGraphNameAllocated(allocator).get();
  } catch (const Ort::Exception& e) {
    return Internal(std::string("onnxruntime: ") + e.what());
  }
  return Status::Ok();
}

Status OnnxRuntimeBackend::Infer(const TensorMap& inputs, TensorMap* outputs) {
  try {
    std::vector<Ort::Value> values;
    values.reserve(input_names_.size());
    for (size_t i = 0; i < input_names_.size(); ++i) {
      auto it = inputs.find(input_names_[i]);
      if (it == inputs.end()) return InvalidArgument("missing model input '" + input_names_[i] + "'");
      const Tensor& t = it->second;
      if (t.dtype() != metadata_.inputs[i].dtype) {
        return InvalidArgument("input '" + input_names_[i] + "' expects " + DataTypeName(metadata_.inputs[i].dtype) +
                               " but got " + DataTypeName(t.dtype()));
      }
      // Zero-copy: ORT reads directly from the batched host buffer.
      values.push_back(Ort::Value::CreateTensor(*cpu_memory_, const_cast<void*>(t.data()), t.byte_size(),
                                                t.shape().data(), t.shape().size(), ToOrtType(t.dtype())));
    }

    auto results = session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(), values.data(), values.size(),
                                 output_name_ptrs_.data(), output_name_ptrs_.size());

    for (size_t i = 0; i < results.size(); ++i) {
      auto info = results[i].GetTensorTypeAndShapeInfo();
      const DataType dtype = FromOrtType(info.GetElementType());
      Shape shape = info.GetShape();
      Tensor out(dtype, shape);
      if (out.byte_size()) {
        std::memcpy(out.data(), results[i].GetTensorMutableData<uint8_t>(), out.byte_size());
      }
      (*outputs)[output_names_[i]] = std::move(out);
    }
  } catch (const Ort::Exception& e) {
    return Internal(std::string("onnxruntime: ") + e.what());
  }
  return Status::Ok();
}

}  // namespace mie

#endif  // MIE_WITH_ONNXRUNTIME
