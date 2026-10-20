// Conversions between engine types and protobuf / gRPC types.
#pragma once

#include <grpcpp/support/status.h>

#include "inference.pb.h"
#include "mie/common/status.h"
#include "mie/common/tensor.h"
#include "mie/engine/model_config.h"

namespace mie {

v1::DataType ToProto(DataType dtype);
DataType FromProto(v1::DataType dtype);

// Copies a proto tensor into an engine tensor, validating its byte size.
Status FromProto(const v1::Tensor& proto, Tensor* tensor);
void ToProto(const std::string& name, const Tensor& tensor, v1::Tensor* proto);
void ToProto(const TensorSpec& spec, v1::TensorSpec* proto);

grpc::Status ToGrpcStatus(const Status& status);

}  // namespace mie
