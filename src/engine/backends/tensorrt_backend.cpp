#include "mie/engine/backends/tensorrt_backend.h"

#if defined(MIE_WITH_TENSORRT) && defined(MIE_WITH_CUDA)

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iterator>

#include "mie/common/logging.h"
#include "mie/memory/cuda_kernels.h"

namespace mie {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR: MIE_LOG(ERROR) << "[TensorRT] " << msg; break;
      case Severity::kWARNING: MIE_LOG(WARNING) << "[TensorRT] " << msg; break;
      case Severity::kINFO: MIE_LOG(DEBUG) << "[TensorRT] " << msg; break;
      default: break;
    }
  }
};

TrtLogger& Logger() {
  static TrtLogger* logger = new TrtLogger();
  return *logger;
}

size_t TrtTypeSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 0;
  }
}

// Client-facing dtype for an engine tensor. Half tensors are exposed as float32.
DataType ClientType(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kHALF: return DataType::kFloat32;
    case nvinfer1::DataType::kINT32: return DataType::kInt32;
    case nvinfer1::DataType::kINT64: return DataType::kInt64;
    case nvinfer1::DataType::kUINT8: return DataType::kUint8;
    case nvinfer1::DataType::kBOOL: return DataType::kBool;
    default: return DataType::kInvalid;
  }
}

Shape DimsToShape(const nvinfer1::Dims& d) { return Shape(d.d, d.d + d.nbDims); }

StatusOr<nvinfer1::Dims> ShapeToDims(const Shape& s) {
  if (s.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
    return InvalidArgument("tensor rank " + std::to_string(s.size()) + " exceeds TensorRT limit");
  }
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(s.size());
  for (size_t i = 0; i < s.size(); ++i) d.d[i] = s[i];
  return d;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool ReadFile(const std::string& path, std::vector<char>* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return !out->empty();
}

}  // namespace

TensorRTBackend::TensorRTBackend() = default;

TensorRTBackend::~TensorRTBackend() {
  // Destroy in dependency order: context -> engine -> runtime.
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

Status TensorRTBackend::BuildEngineFromOnnx(const ModelConfig& config, std::vector<char>* plan) {
  using namespace nvinfer1;
  std::unique_ptr<IBuilder> builder(createInferBuilder(Logger()));
  if (!builder) return Internal("createInferBuilder failed");
  // TensorRT 10 networks are always explicit-batch.
  std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, Logger()));
  if (!parser->parseFromFile(config.path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
    std::string errors;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) errors += std::string(parser->getError(i)->desc()) + "; ";
    return InvalidArgument("failed to parse ONNX model " + config.path + ": " + errors);
  }

  std::unique_ptr<IBuilderConfig> builder_config(builder->createBuilderConfig());
  builder_config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,
                                     static_cast<size_t>(config.GetIntParameter("workspace_bytes", int64_t{1} << 30)));
  if (config.GetBoolParameter("fp16", true)) {
    // kFP16 is deprecated in newer TensorRT 10 releases in favour of strongly
    // typed networks, but it is still the way to let the builder pick FP16
    // kernels for an FP32 ONNX model.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    builder_config->setFlag(BuilderFlag::kFP16);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  }

  const int64_t max_batch = config.GetIntParameter("max_batch", std::max(1, config.batching.max_batch_size));
  const int64_t opt_batch = config.GetIntParameter("opt_batch", max_batch);
  const int64_t min_batch = config.GetIntParameter("min_batch", 1);

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  bool has_dynamic = false;
  for (int32_t i = 0; i < network->getNbInputs(); ++i) {
    ITensor* input = network->getInput(i);
    const Dims dims = input->getDimensions();
    Dims mn = dims, opt = dims, mx = dims;
    bool dynamic = false;
    for (int32_t j = 0; j < dims.nbDims; ++j) {
      if (dims.d[j] >= 0) continue;
      dynamic = true;
      if (j == 0) {
        mn.d[0] = min_batch;
        opt.d[0] = opt_batch;
        mx.d[0] = max_batch;
        continue;
      }
      // Non-batch dynamic dims must be pinned through the model config.
      const TensorSpec* spec = nullptr;
      for (const auto& s : config.inputs) {
        if (s.name == input->getName()) spec = &s;
      }
      if (!spec || static_cast<int32_t>(spec->shape.size()) != dims.nbDims || spec->shape[j] <= 0) {
        return InvalidArgument(std::string("input '") + input->getName() + "' has dynamic dim " + std::to_string(j) +
                               "; specify a concrete shape in the model config inputs");
      }
      mn.d[j] = opt.d[j] = mx.d[j] = spec->shape[j];
    }
    if (dynamic) {
      has_dynamic = true;
      profile->setDimensions(input->getName(), OptProfileSelector::kMIN, mn);
      profile->setDimensions(input->getName(), OptProfileSelector::kOPT, opt);
      profile->setDimensions(input->getName(), OptProfileSelector::kMAX, mx);
    }
  }
  if (has_dynamic) builder_config->addOptimizationProfile(profile);

  MIE_LOG(INFO) << "building TensorRT engine for " << config.path << " (fp16=" << config.GetBoolParameter("fp16", true)
                << ", batch " << min_batch << "/" << opt_batch << "/" << max_batch << "); this can take a while";
  std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *builder_config));
  if (!serialized || serialized->size() == 0) return Internal("TensorRT engine build failed");
  const auto* bytes = static_cast<const char*>(serialized->data());
  plan->assign(bytes, bytes + serialized->size());
  return Status::Ok();
}

Status TensorRTBackend::InitBindings(const ModelConfig& config) {
  bindings_.clear();
  metadata_ = ModelMetadata{};
  bool dynamic_batch = true;
  int max_batch = std::max(1, config.batching.max_batch_size);
  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    IoBinding b;
    b.name = engine_->getIOTensorName(i);
    b.is_input = engine_->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    b.engine_dtype = engine_->getTensorDataType(b.name.c_str());
    b.client_dtype = ClientType(b.engine_dtype);
    if (b.client_dtype == DataType::kInvalid || TrtTypeSize(b.engine_dtype) == 0) {
      return Unimplemented("tensor '" + b.name + "' uses an unsupported TensorRT data type");
    }
    const Shape shape = DimsToShape(engine_->getTensorShape(b.name.c_str()));
    if (b.is_input) {
      if (shape.empty() || shape[0] >= 0) {
        dynamic_batch = false;
      } else {
        const auto mx = engine_->getProfileShape(b.name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        if (mx.nbDims > 0) max_batch = std::min<int>(max_batch, static_cast<int>(mx.d[0]));
      }
      metadata_.inputs.push_back({b.name, b.client_dtype, shape});
    } else {
      metadata_.outputs.push_back({b.name, b.client_dtype, shape});
    }
    bindings_.push_back(std::move(b));
  }
  metadata_.max_batch_size = dynamic_batch ? max_batch : 0;
  metadata_.platform = "tensorrt-" + std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) +
                       "." + std::to_string(NV_TENSORRT_PATCH);
  return Status::Ok();
}

Status TensorRTBackend::Load(const ModelConfig& config) {
  if (config.path.empty()) return InvalidArgument("tensorrt backend requires a model path");
  device_id_ = config.device_id;
  MIE_CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id_));
  MIE_ASSIGN_OR_RETURN(device_pool_, DeviceMemoryPools::Instance().Get(device_id_));

  std::vector<char> plan;
  if (EndsWith(config.path, ".engine") || EndsWith(config.path, ".plan")) {
    if (!ReadFile(config.path, &plan)) return NotFound("cannot read TensorRT engine " + config.path);
  } else {
    const bool fp16 = config.GetBoolParameter("fp16", true);
    const std::string cache = config.GetParameter("engine_cache", config.path + (fp16 ? ".fp16" : ".fp32") + ".engine");
    const bool rebuild = config.GetBoolParameter("rebuild_engine", false);
    if (!rebuild && ReadFile(cache, &plan)) {
      MIE_LOG(INFO) << "using cached TensorRT engine " << cache;
    } else {
      MIE_RETURN_IF_ERROR(BuildEngineFromOnnx(config, &plan));
      std::ofstream out(cache, std::ios::binary);
      if (out.write(plan.data(), static_cast<std::streamsize>(plan.size()))) {
        MIE_LOG(INFO) << "cached TensorRT engine at " << cache;
      } else {
        MIE_LOG(WARNING) << "could not write engine cache " << cache;
      }
    }
  }

  runtime_.reset(nvinfer1::createInferRuntime(Logger()));
  if (!runtime_) return Internal("createInferRuntime failed");
  engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine_) return Internal("failed to deserialize TensorRT engine (built for a different GPU/TensorRT version?)");
  context_.reset(engine_->createExecutionContext());
  if (!context_) return Internal("createExecutionContext failed");
  stream_ = std::make_unique<CudaStream>(device_id_);
  if (!stream_->get()) return Internal("failed to create CUDA stream");
  MIE_RETURN_IF_ERROR(InitBindings(config));
  metadata_.properties["device_id"] = std::to_string(device_id_);
  metadata_.properties["fp16"] = config.GetBoolParameter("fp16", true) ? "1" : "0";
  return Status::Ok();
}

Status TensorRTBackend::Infer(const TensorMap& inputs, TensorMap* outputs) {
  std::lock_guard<std::mutex> lock(mu_);
  MIE_CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id_));
  cudaStream_t stream = stream_->get();
  // Keeps every device buffer used by this batch alive until the stream syncs.
  std::vector<PooledBuffer> buffers;
  buffers.reserve(bindings_.size() * 2);
  // Declared after `buffers` so it runs first on every exit path: blocks must
  // not return to the pool while the GPU may still be using them.
  struct SyncOnExit {
    cudaStream_t s;
    ~SyncOnExit() { cudaStreamSynchronize(s); }
  } sync_guard{stream};

  // 1) Inputs: set shapes, upload (converting to half on-device if needed).
  for (const auto& b : bindings_) {
    if (!b.is_input) continue;
    auto it = inputs.find(b.name);
    if (it == inputs.end()) return InvalidArgument("missing model input '" + b.name + "'");
    const Tensor& t = it->second;
    if (t.dtype() != b.client_dtype) {
      return InvalidArgument("input '" + b.name + "' expects " + DataTypeName(b.client_dtype) + ", got " +
                             DataTypeName(t.dtype()));
    }
    MIE_ASSIGN_OR_RETURN(nvinfer1::Dims dims, ShapeToDims(t.shape()));
    if (!context_->setInputShape(b.name.c_str(), dims)) {
      return InvalidArgument("shape " + ShapeToString(t.shape()) + " is outside the engine profile for '" + b.name + "'");
    }
    const int64_t count = t.num_elements();
    MIE_ASSIGN_OR_RETURN(PooledBuffer device, device_pool_->Acquire(static_cast<size_t>(count) * TrtTypeSize(b.engine_dtype)));
    if (b.engine_dtype == nvinfer1::DataType::kHALF) {
      MIE_ASSIGN_OR_RETURN(PooledBuffer staging, device_pool_->Acquire(t.byte_size()));
      MIE_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(staging.data(), t.data(), t.byte_size(), cudaMemcpyHostToDevice, stream));
      MIE_CUDA_RETURN_IF_ERROR(cuda::CastFloatToHalf(staging.as<float>(), device.data(), count, stream));
      buffers.push_back(std::move(staging));
    } else {
      MIE_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device.data(), t.data(), t.byte_size(), cudaMemcpyHostToDevice, stream));
    }
    if (!context_->setTensorAddress(b.name.c_str(), device.data())) {
      return Internal("setTensorAddress failed for '" + b.name + "'");
    }
    buffers.push_back(std::move(device));
  }

  // 2) Outputs: shapes are known once all input shapes are set.
  struct PendingOutput {
    const IoBinding* binding;
    void* device;
    int64_t count;
  };
  std::vector<PendingOutput> pending;
  for (const auto& b : bindings_) {
    if (b.is_input) continue;
    const Shape shape = DimsToShape(context_->getTensorShape(b.name.c_str()));
    const int64_t count = NumElements(shape);
    if (count < 0) return Internal("unresolved output shape for '" + b.name + "'");
    MIE_ASSIGN_OR_RETURN(PooledBuffer device,
                         device_pool_->Acquire(std::max<size_t>(1, static_cast<size_t>(count) * TrtTypeSize(b.engine_dtype))));
    if (!context_->setTensorAddress(b.name.c_str(), device.data())) {
      return Internal("setTensorAddress failed for '" + b.name + "'");
    }
    (*outputs)[b.name] = Tensor(b.client_dtype, shape);
    pending.push_back({&b, device.data(), count});
    buffers.push_back(std::move(device));
  }

  // 3) Execute.
  if (!context_->enqueueV3(stream)) return Internal("TensorRT enqueueV3 failed");

  // 4) Download (converting half -> float on-device first).
  for (const auto& p : pending) {
    Tensor& host = (*outputs)[p.binding->name];
    const void* src = p.device;
    if (p.binding->engine_dtype == nvinfer1::DataType::kHALF) {
      MIE_ASSIGN_OR_RETURN(PooledBuffer converted, device_pool_->Acquire(host.byte_size()));
      MIE_CUDA_RETURN_IF_ERROR(cuda::CastHalfToFloat(p.device, converted.as<float>(), p.count, stream));
      src = converted.data();
      buffers.push_back(std::move(converted));
    }
    if (host.byte_size()) {
      MIE_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host.data(), src, host.byte_size(), cudaMemcpyDeviceToHost, stream));
    }
  }
  return stream_->Synchronize();
}

}  // namespace mie

#endif  // MIE_WITH_TENSORRT && MIE_WITH_CUDA
