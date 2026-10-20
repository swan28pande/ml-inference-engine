#include <gtest/gtest.h>

#include "mie/common/tensor.h"

namespace mie {
namespace {

TEST(TensorTest, ConstructsZeroFilled) {
  Tensor t(DataType::kFloat32, {2, 3});
  EXPECT_EQ(t.byte_size(), 24u);
  EXPECT_EQ(t.num_elements(), 6);
  EXPECT_EQ(t.batch_size(), 2);
  EXPECT_EQ(t.sample_byte_size(), 12u);
  for (float v : t.ToVector<float>()) EXPECT_EQ(v, 0.0f);
  EXPECT_TRUE(t.Validate().ok());
}

TEST(TensorTest, ValidateDetectsSizeMismatch) {
  Tensor t(DataType::kFloat32, {2, 3}, std::vector<uint8_t>(10));
  EXPECT_EQ(t.Validate().code(), StatusCode::kInvalidArgument);
}

TEST(TensorTest, DataTypeRoundTrip) {
  for (DataType dt : {DataType::kFloat32, DataType::kFloat16, DataType::kInt64, DataType::kInt32, DataType::kUint8,
                      DataType::kBool}) {
    EXPECT_EQ(DataTypeFromString(DataTypeName(dt)), dt);
  }
  EXPECT_EQ(DataTypeFromString("FP16"), DataType::kFloat16);
  EXPECT_EQ(DataTypeFromString("nope"), DataType::kInvalid);
}

TEST(TensorTest, ConcatAndSplitRoundTrip) {
  Tensor a = Tensor::FromVector<float>({1, 2}, {1, 2});
  Tensor b = Tensor::FromVector<float>({3, 2}, {3, 4, 5, 6, 7, 8});
  Tensor c = Tensor::FromVector<float>({2, 2}, {9, 10, 11, 12});

  Tensor merged;
  ASSERT_TRUE(ConcatBatch({&a, &b, &c}, &merged).ok());
  EXPECT_EQ(merged.shape(), (Shape{6, 2}));
  EXPECT_EQ(merged.ToVector<float>(), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  std::vector<Tensor> parts;
  ASSERT_TRUE(SplitBatch(merged, {1, 3, 2}, &parts).ok());
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_EQ(parts[0].ToVector<float>(), a.ToVector<float>());
  EXPECT_EQ(parts[1].ToVector<float>(), b.ToVector<float>());
  EXPECT_EQ(parts[2].ToVector<float>(), c.ToVector<float>());
  EXPECT_EQ(parts[1].shape(), (Shape{3, 2}));
}

TEST(TensorTest, ConcatRejectsMismatchedSamples) {
  Tensor a(DataType::kFloat32, {1, 2});
  Tensor b(DataType::kFloat32, {1, 3});
  Tensor c(DataType::kInt64, {1, 2});
  Tensor out;
  EXPECT_FALSE(ConcatBatch({&a, &b}, &out).ok());
  EXPECT_FALSE(ConcatBatch({&a, &c}, &out).ok());
  EXPECT_FALSE(ConcatBatch({}, &out).ok());
}

TEST(TensorTest, SplitRejectsWrongSizes) {
  Tensor t(DataType::kFloat32, {4, 2});
  std::vector<Tensor> parts;
  EXPECT_FALSE(SplitBatch(t, {1, 2}, &parts).ok());
}

TEST(TensorTest, MultiDimensionalSamples) {
  std::vector<float> va(1 * 2 * 3), vb(2 * 2 * 3);
  for (size_t i = 0; i < va.size(); ++i) va[i] = static_cast<float>(i);
  for (size_t i = 0; i < vb.size(); ++i) vb[i] = 100.0f + static_cast<float>(i);
  Tensor a = Tensor::FromVector<float>({1, 2, 3}, va);
  Tensor b = Tensor::FromVector<float>({2, 2, 3}, vb);
  Tensor merged;
  ASSERT_TRUE(ConcatBatch({&a, &b}, &merged).ok());
  EXPECT_EQ(merged.shape(), (Shape{3, 2, 3}));
  std::vector<Tensor> parts;
  ASSERT_TRUE(SplitBatch(merged, {1, 2}, &parts).ok());
  EXPECT_EQ(parts[1].ToVector<float>(), vb);
}

}  // namespace
}  // namespace mie
