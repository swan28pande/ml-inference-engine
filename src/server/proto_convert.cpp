#include "mie/server/proto_convert.h"

namespace mie {

v1::DataType ToProto(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32: return v1::DT_FLOAT32;
    case DataType::kFloat16: return v1::DT_FLOAT16;
    case DataType::kInt64: return v1::DT_INT64;
    case DataType::kInt32: return v1::DT_INT32;
    case DataType::kUint8: return v1::DT_UINT8;
    case DataType::kBool: return v1::DT_BOOL;
    case DataType::kInvalid: break;
  }
  return v1::DT_INVALID;
}

DataType FromProto(v1::DataType dtype) {
  switch (dtype) {
    case v1::DT_FLOAT32: return DataType::kFloat32;
    case v1::DT_FLOAT16: return DataType::kFloat16;
    case v1::DT_INT64: return DataType::kInt64;
    case v1::DT_INT32: return DataType::kInt32;
    case v1::DT_UINT8: return DataType::kUint8;
    case v1::DT_BOOL: return DataType::kBool;
    default: return DataType::kInvalid;
  }
}

Status FromProto(const v1::Tensor& proto, Tensor* tensor) {
  const DataType dtype = FromProto(proto.dtype());
  if (dtype == DataType::kInvalid) return InvalidArgument("tensor '" + proto.name() + "' has an invalid dtype");
  Shape shape(proto.shape().begin(), proto.shape().end());
  const std::string& raw = proto.raw_data();
  std::vector<uint8_t> bytes(raw.begin(), raw.end());
  *tensor = Tensor(dtype, std::move(shape), std::move(bytes));
  Status s = tensor->Validate();
  if (!s.ok()) return InvalidArgument("tensor '" + proto.name() + "': " + s.message());
  return Status::Ok();
}

void ToProto(const std::string& name, const Tensor& tensor, v1::Tensor* proto) {
  proto->set_name(name);
  proto->set_dtype(ToProto(tensor.dtype()));
  for (int64_t d : tensor.shape()) proto->add_shape(d);
  proto->set_raw_data(tensor.data(), tensor.byte_size());
}

void ToProto(const TensorSpec& spec, v1::TensorSpec* proto) {
  proto->set_name(spec.name);
  proto->set_dtype(ToProto(spec.dtype));
  for (int64_t d : spec.shape) proto->add_shape(d);
}

grpc::Status ToGrpcStatus(const Status& status) {
  using grpc::StatusCode;
  StatusCode code = StatusCode::UNKNOWN;
  switch (status.code()) {
    case mie::StatusCode::kOk: return grpc::Status::OK;
    case mie::StatusCode::kInvalidArgument: code = StatusCode::INVALID_ARGUMENT; break;
    case mie::StatusCode::kNotFound: code = StatusCode::NOT_FOUND; break;
    case mie::StatusCode::kAlreadyExists: code = StatusCode::ALREADY_EXISTS; break;
    case mie::StatusCode::kFailedPrecondition: code = StatusCode::FAILED_PRECONDITION; break;
    case mie::StatusCode::kResourceExhausted: code = StatusCode::RESOURCE_EXHAUSTED; break;
    case mie::StatusCode::kUnavailable: code = StatusCode::UNAVAILABLE; break;
    case mie::StatusCode::kDeadlineExceeded: code = StatusCode::DEADLINE_EXCEEDED; break;
    case mie::StatusCode::kCancelled: code = StatusCode::CANCELLED; break;
    case mie::StatusCode::kUnimplemented: code = StatusCode::UNIMPLEMENTED; break;
    case mie::StatusCode::kInternal: code = StatusCode::INTERNAL; break;
  }
  return grpc::Status(code, status.message());
}

}  // namespace mie
