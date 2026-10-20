#include "mie/common/tensor.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace mie {

size_t DataTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32: return 4;
    case DataType::kFloat16: return 2;
    case DataType::kInt64: return 8;
    case DataType::kInt32: return 4;
    case DataType::kUint8: return 1;
    case DataType::kBool: return 1;
    case DataType::kInvalid: return 0;
  }
  return 0;
}

const char* DataTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32: return "float32";
    case DataType::kFloat16: return "float16";
    case DataType::kInt64: return "int64";
    case DataType::kInt32: return "int32";
    case DataType::kUint8: return "uint8";
    case DataType::kBool: return "bool";
    case DataType::kInvalid: return "invalid";
  }
  return "invalid";
}

DataType DataTypeFromString(const std::string& name) {
  std::string n(name);
  std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
  if (n == "float32" || n == "float" || n == "fp32") return DataType::kFloat32;
  if (n == "float16" || n == "half" || n == "fp16") return DataType::kFloat16;
  if (n == "int64") return DataType::kInt64;
  if (n == "int32") return DataType::kInt32;
  if (n == "uint8") return DataType::kUint8;
  if (n == "bool") return DataType::kBool;
  return DataType::kInvalid;
}

int64_t NumElements(const Shape& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) return -1;
    n *= d;
  }
  return n;
}

std::string ShapeToString(const Shape& shape) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ',';
    os << shape[i];
  }
  os << ']';
  return os.str();
}

Tensor::Tensor(DataType dtype, Shape shape) : dtype_(dtype), shape_(std::move(shape)) {
  const int64_t n = NumElements(shape_);
  data_.assign(n > 0 ? static_cast<size_t>(n) * DataTypeSize(dtype_) : 0, 0);
}

Tensor::Tensor(DataType dtype, Shape shape, std::vector<uint8_t> bytes)
    : dtype_(dtype), shape_(std::move(shape)), data_(std::move(bytes)) {}

Status Tensor::Validate() const {
  if (dtype_ == DataType::kInvalid) return InvalidArgument("tensor has invalid dtype");
  const int64_t n = NumElements(shape_);
  if (n < 0) return InvalidArgument("tensor shape " + ShapeToString(shape_) + " has negative dimensions");
  const size_t expected = static_cast<size_t>(n) * DataTypeSize(dtype_);
  if (expected != data_.size()) {
    return InvalidArgument("tensor with shape " + ShapeToString(shape_) + " and dtype " + DataTypeName(dtype_) +
                           " expects " + std::to_string(expected) + " bytes but has " +
                           std::to_string(data_.size()));
  }
  return Status::Ok();
}

size_t Tensor::sample_byte_size() const {
  if (shape_.empty()) return data_.size();
  size_t n = DataTypeSize(dtype_);
  for (size_t i = 1; i < shape_.size(); ++i) n *= static_cast<size_t>(shape_[i]);
  return n;
}

bool SameSampleSignature(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype()) return false;
  const Shape& sa = a.shape();
  const Shape& sb = b.shape();
  if (sa.size() != sb.size() || sa.empty()) return false;
  return std::equal(sa.begin() + 1, sa.end(), sb.begin() + 1);
}

Status ConcatBatch(const std::vector<const Tensor*>& parts, Tensor* out) {
  if (parts.empty()) return InvalidArgument("ConcatBatch: no inputs");
  const Tensor& first = *parts.front();
  if (first.shape().empty()) return InvalidArgument("ConcatBatch: scalar tensors cannot be batched");

  int64_t total = 0;
  size_t bytes = 0;
  for (const Tensor* t : parts) {
    if (!SameSampleSignature(first, *t)) {
      return InvalidArgument("ConcatBatch: incompatible shapes " + ShapeToString(first.shape()) + " vs " +
                             ShapeToString(t->shape()));
    }
    total += t->shape()[0];
    bytes += t->byte_size();
  }

  Shape shape = first.shape();
  shape[0] = total;
  std::vector<uint8_t> data(bytes);
  size_t offset = 0;
  for (const Tensor* t : parts) {
    if (t->byte_size()) std::memcpy(data.data() + offset, t->data(), t->byte_size());
    offset += t->byte_size();
  }
  *out = Tensor(first.dtype(), std::move(shape), std::move(data));
  return Status::Ok();
}

Status SplitBatch(const Tensor& batched, const std::vector<int64_t>& sizes, std::vector<Tensor>* out) {
  if (batched.shape().empty()) return InvalidArgument("SplitBatch: scalar tensor");
  int64_t total = 0;
  for (int64_t s : sizes) total += s;
  if (total != batched.shape()[0]) {
    return InvalidArgument("SplitBatch: sizes sum to " + std::to_string(total) + " but batch dim is " +
                           std::to_string(batched.shape()[0]));
  }
  const size_t sample_bytes = batched.sample_byte_size();
  out->clear();
  out->reserve(sizes.size());
  const auto* src = static_cast<const uint8_t*>(batched.data());
  size_t offset = 0;
  for (int64_t s : sizes) {
    Shape shape = batched.shape();
    shape[0] = s;
    const size_t n = sample_bytes * static_cast<size_t>(s);
    std::vector<uint8_t> data(src + offset, src + offset + n);
    offset += n;
    out->emplace_back(batched.dtype(), std::move(shape), std::move(data));
  }
  return Status::Ok();
}

}  // namespace mie
