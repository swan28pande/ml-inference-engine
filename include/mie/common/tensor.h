// Host tensor type and batch concatenation / splitting helpers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "mie/common/status.h"

namespace mie {

enum class DataType : uint8_t {
  kInvalid = 0,
  kFloat32,
  kFloat16,
  kInt64,
  kInt32,
  kUint8,
  kBool,
};

size_t DataTypeSize(DataType dtype);
const char* DataTypeName(DataType dtype);
DataType DataTypeFromString(const std::string& name);

template <typename T> struct DataTypeOf;
template <> struct DataTypeOf<float>   { static constexpr DataType value = DataType::kFloat32; };
template <> struct DataTypeOf<int64_t> { static constexpr DataType value = DataType::kInt64; };
template <> struct DataTypeOf<int32_t> { static constexpr DataType value = DataType::kInt32; };
template <> struct DataTypeOf<uint8_t> { static constexpr DataType value = DataType::kUint8; };

using Shape = std::vector<int64_t>;

// Product of all dimensions. Returns -1 if any dimension is dynamic (< 0).
int64_t NumElements(const Shape& shape);
std::string ShapeToString(const Shape& shape);

// A dense, row-major host tensor that owns its storage. Dimension 0 is the
// batch dimension for all tensors flowing through the scheduler.
class Tensor {
 public:
  Tensor() = default;
  Tensor(DataType dtype, Shape shape);  // zero-initialised
  Tensor(DataType dtype, Shape shape, std::vector<uint8_t> bytes);

  template <typename T>
  static Tensor FromVector(Shape shape, const std::vector<T>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return Tensor(DataTypeOf<T>::value, std::move(shape), std::move(bytes));
  }

  template <typename T>
  std::vector<T> ToVector() const {
    std::vector<T> out(data_.size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), data_.data(), out.size() * sizeof(T));
    return out;
  }

  DataType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  int64_t num_elements() const { return NumElements(shape_); }
  size_t byte_size() const { return data_.size(); }
  int64_t batch_size() const { return shape_.empty() ? 1 : shape_[0]; }

  void* data() { return data_.data(); }
  const void* data() const { return data_.data(); }
  template <typename T> T* data_as() { return reinterpret_cast<T*>(data_.data()); }
  template <typename T> const T* data_as() const { return reinterpret_cast<const T*>(data_.data()); }

  std::vector<uint8_t>& mutable_bytes() { return data_; }
  const std::vector<uint8_t>& bytes() const { return data_; }

  // Checks that the byte size matches dtype and shape.
  Status Validate() const;

  // Bytes of one sample (all dims except the batch dim).
  size_t sample_byte_size() const;

 private:
  DataType dtype_ = DataType::kInvalid;
  Shape shape_;
  std::vector<uint8_t> data_;
};

using TensorMap = std::map<std::string, Tensor>;

// True if both tensors share dtype and all non-batch dimensions.
bool SameSampleSignature(const Tensor& a, const Tensor& b);

// Concatenates tensors along dimension 0. All parts must share dtype and
// non-batch dimensions.
Status ConcatBatch(const std::vector<const Tensor*>& parts, Tensor* out);

// Splits a tensor along dimension 0 into pieces of the given batch sizes.
// The sizes must sum to tensor.shape()[0].
Status SplitBatch(const Tensor& batched, const std::vector<int64_t>& sizes, std::vector<Tensor>* out);

}  // namespace mie
