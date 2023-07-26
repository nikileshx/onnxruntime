// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <cmath>  // NAN
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

namespace {
constexpr char kOpName[] = "IsNan";
constexpr int kOpVersion = 9;

template <typename TTarget, typename TNarrow = int8_t>
void NonZeroBasicNumericTest() {
  OpTester test{kOpName, kOpVersion};

  std::vector<int64_t> X_dims{1, 2, 3};
  std::vector<TNarrow> X{0, 1, 2,
                         0, 3, 4};
  test.AddInput<TTarget>("X", X_dims, std::vector<TTarget>{X.begin(), X.end()});
  test.AddOutput<int64_t>(
      "Y", {3, 4},
      {0, 0, 0, 0,
       0, 0, 1, 1,
       1, 2, 1, 2});
  test.Run();
}
}  // namespace

TEST(IsNanTest, BasicNumeric) {
  NonZeroBasicNumericTest<int32_t>();
  NonZeroBasicNumericTest<int64_t>();
  NonZeroBasicNumericTest<float>();
}
// TEST(IsNaNOpTest, IsNaNFloat16) {
//   OpTester test("IsNaN", 9, kOnnxDomain);
//   std::vector<int64_t> dims{2, 2};
//   test.AddInput<MLFloat16>("X", dims, std::initializer_list<MLFloat16>({MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(NAN)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(NAN))}));
//   test.AddOutput<bool>("Y", dims, {false, true, false, true});
//   test.Run();
// }

// TEST(IsNaNOpTest, IsNaNDouble) {
//   OpTester test("IsNaN", 9, kOnnxDomain);
//   std::vector<int64_t> dims{2, 2};
//   test.AddInput<double>("X", dims, {1.0, NAN, 2.0, NAN});
//   test.AddOutput<bool>("Y", dims, {false, true, false, true});
//   test.Run();
// }

}  // namespace test
}  // namespace onnxruntime
