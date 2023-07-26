// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <core/tensorview/tensorview.h>
#include "onnxruntime_cxx_api.h"
// #include <torch/script.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>

namespace tv {

// struct TorchGPU: public tv::GPU {
//   virtual cudaStream_t getStream() const override {
//     return at::cuda::getCurrentCUDAStream();
//   }
// };

// template <typename T> void check_torch_dtype(const torch::Tensor &tensor) {
//   switch (tensor.type().scalarType()) {
//   case at::ScalarType::Double: {
//     auto val = std::is_same<std::remove_const_t<T>, double>::value;
//     TV_ASSERT_RT_ERR(val, "error");
//     break;
//   }
//   case at::ScalarType::Float: {
//     auto val = std::is_same<std::remove_const_t<T>, float>::value;
//     TV_ASSERT_RT_ERR(val, "error");
//     break;
//   }
//   case at::ScalarType::Int: {
//     auto val = std::is_same<std::remove_const_t<T>, int>::value;
//     TV_ASSERT_RT_ERR(val, "error");
//     break;
//   }
//   case at::ScalarType::Half: {
//     auto val = std::is_same<std::remove_const_t<T>, at::Half>::value;
//     TV_ASSERT_RT_ERR(val, "error");
//     break;
//   }
//   case at::ScalarType::Long: {
//     auto val = std::is_same<std::remove_const_t<T>, long>::value;
//     TV_ASSERT_RT_ERR(val, "error");
//     break;
//   }
//   default:
//     TV_ASSERT_RT_ERR(false, "error");
//   }
// }

// template <typename T>
// tv::TensorView<T> ort2tv(const Ort::Value &tensor) {
//   using MutableT = std::remove_const_t<T>;
//   std::cout << "INSIDE tv::Ort2tv() tensor argument \n";
//   tv::Shape shape;
//   auto tensorShape = tensor.GetTensorTypeAndShapeInfo().GetShape();
//   for (size_t i = 0; i < tensorShape.size(); ++i) {
//     shape.push_back(tensorShape[i]);
//   }
//   std::cout << "Returning from tv::Ort2tv() \n";
//   // return tv::TensorView<T>(tensor.GetTensorRawData<T>(), shape);
//   return tv::TensorView<MutableT>(tensor.GetTensorRawData<MutableT>(), shape);
// }


template <typename T>
tv::TensorView<T> ort2tv(Ort::Value &tensor) {
  // check_torch_dtype<T>(tensor);
  std::cout << "INSIDE tv::Ort2tv() tensor argument \n";
  tv::Shape shape;
  auto tensorShape = tensor.GetTensorTypeAndShapeInfo().GetShape();
  for (size_t i = 0; i < tensorShape.size(); ++i) {
    shape.push_back(tensorShape[i]);
  }
  std::cout << "Returning from tv::Ort2tv() \n";
  // if tensor is const then return this ....
  return tv::TensorView<T>(tensor.GetTensorMutableData<T>(), shape);
}

template <typename T>
tv::TensorView<T> ort2tv(const T* tensorData, const std::vector<int64_t> tensorShape) {
  // check_torch_dtype<T>(tensor);
  std::cout << "INSIDE tv::Ort2tv() tensorData argument \n";
  tv::Shape shape;
//   auto tensorShape = tensor.GetTensorTypeAndShapeInfo().GetShape();
  for (size_t i = 0; i < tensorShape.size(); ++i) {
    shape.push_back(tensorShape[i]);
  }
  std::cout << "Returning from tv::Ort2tv() tensorData argument \n";
  return tv::TensorView<T>(tensorData, shape);
  // return tv::TensorView<T>(tensor.GetTensorData<T>(), shape);
}

// template <typename T>
// tv::TensorView<const T> ort2tv(const Ort::Value& tensor) {
//   std::cout << "INSIDE tv::Ort2tv() const \n";
//   tv::Shape shape;
//   auto tensorShape = tensor.GetTensorTypeAndShapeInfo().GetShape();
//   for (size_t i = 0; i < tensorShape.size(); ++i) {
//     shape.push_back(tensorShape[i]);
//   }
//   std::cout << "Returning from tv::Ort2tv() \n";
//   return tv::TensorView<const T>(tensor.GetTensorData<T>(), shape);
// }


// template <typename T>
// struct ort2tv_helper {
//   static tv::TensorView<T> convert(const Ort::Value& tensor) {
//     std::cout << "INSIDE STANDARD ort2tv Impl\n";
//     tv::Shape shape;
//     auto tensorShape = tensor.GetTensorTypeAndShapeInfo().GetShape();
//     for (size_t i = 0; i < tensorShape.size(); ++i) {
//       shape.push_back(tensorShape[i]);
//     }
//     return tv::TensorView<T>(tensor.GetTensorMutableData<T>(), shape);
//   }
// };

// template <>
// struct ort2tv_helper<const int64_t> {
//   static tv::TensorView<const int64_t> convert(const OrtValue& tensor) {
//     std::cout << "INSIDE secondary ort2tv Impl\n";
//     tv::Shape shape;
//     auto tensorShape = tensor.GetTensorTypeAndShapeInfo().GetShape();
//     for (size_t i = 0; i < tensorShape.size(); ++i) {
//       shape.push_back(tensorShape[i]);
//     }
//     return tv::TensorView<const int64_t>(tensor.GetTensorData<int64_t>(), shape);
//   }
// };

// template <typename T>
// tv::TensorView<T> ort2tv(const T& tensor) {
//   return ort2tv_helper<T>::convert(tensor);
// }


} // namespace tv
