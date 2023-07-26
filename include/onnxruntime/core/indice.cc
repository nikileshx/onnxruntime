// // Copyright 2019 Yan Yan
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include <core/geometry.h>
// #include <core/indice.h>
// #include <mycustom_op_library/spconv_ops.h>

// namespace functor {
// // template <typename Index, typename IndexGrid, unsigned NDim>
// // template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
// // struct CreateConvIndicePairFunctor {
// //   void operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn) {
// //     std::cout << "INSIDE CreateConvIndicePairFunctor cc implementation\n";
// //     std::cout << "INSIDE REAL FUNCTOR::CreateConvIndicePairFunctor \n";
// //     std::cout << "indicesIn TensorView:\n";
// //     tv::printTensorView(indicesIn);
// //   }
// // };

// // struct CreateConvIndicePairFunctor<tv::CPU, Index, IndexGrid, NDim> {
// //   void operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn) {
// //     std::cout << "INSIDE CreateConvIndicePairFunctor cc implementation\n";
// //     std::cout << "INSIDE REAL FUNCTOR::CreateConvIndicePairFunctor \n";
// //     std::cout << "indicesIn TensorView:\n";
// //     tv::printTensorView(indicesIn);

//   // Index operator()(const tv::CPU& d, tv::TensorView<const Index> indicesIn,
//   //                    tv::TensorView<Index> indicesOut,
//   //                    tv::TensorView<IndexGrid> gridsOut,
//   //                    tv::TensorView<Index> indicePairs,
//   //                    tv::TensorView<Index> indiceNum,
//   //                    const tv::SimpleVector<Index, NDim> kernelSize,
//   //                    const tv::SimpleVector<Index, NDim> stride,
//   //                    const tv::SimpleVector<Index, NDim> padding,
//   //                    const tv::SimpleVector<Index, NDim> dilation,
//   //                    const tv::SimpleVector<Index, NDim> outSpatialShape,
//   //                    bool transpose, bool resetGrid) {
//   //   // if (transpose)
//   //   //   return getIndicePairsDeConv<Index, IndexGrid, NDim>(
//   //   //       indicesIn, indicesOut,
//   //   //       gridsOut, indicePairs, indiceNum,
//   //   //       kernelSize.data(), stride.data(), padding.data(), dilation.data(),
//   //   //       outSpatialShape.data());
//   //   // else
//   //   std::cout << "INSIDE CreateConvIndicePairFunctor cc implementation\n";
//   //   std::cout << "INSIDE REAL FUNCTOR::CreateConvIndicePairFunctor \n";
//   //   std::cout << "indicesIn TensorView:\n";
//   //   tv::printTensorView(indicesIn);
//   //   // return getIndicePairsConv<Index, IndexGrid, NDim>(
//   //   //     indicesIn, indicesOut,
//   //   //     gridsOut, indicePairs, indiceNum,
//   //   //     kernelSize.data(), stride.data(), padding.data(), dilation.data(),
//   //   //     outSpatialShape.data());

// //   }
// // };
// } // namespace functor

// // #define DECLARE_CPU_SPECS_INDEX_NDIM(Index, NDIM)                              \
// //   template struct functor::CreateConvIndicePairFunctor<tv::CPU, Index, int64_t, NDIM>;      \
// // //   template struct functor::CreateSubMIndicePairFunctor<tv::CPU, Index, int,  \
// // //                                                          NDIM>;


// // #define DECLARE_CPU_INDEX(Index)                                               \
// //   DECLARE_CPU_SPECS_INDEX_NDIM(Index, 1);                                      \
// //   DECLARE_CPU_SPECS_INDEX_NDIM(Index, 2);                                      \
// //   DECLARE_CPU_SPECS_INDEX_NDIM(Index, 3);                                      \
// //   DECLARE_CPU_SPECS_INDEX_NDIM(Index, 4);

// // DECLARE_CPU_INDEX(int64_t);
// // // DECLARE_CPU_INDEX(long);

// // #undef DECLARE_CPU_INDEX
// // #undef DECLARE_CPU_SPECS_INDEX_NDIM



// // ---------------------*******************************************----------------------------------

// // #include "core/geometry.h"
// #include "core/indice.h"
// // #include "core/tensorview/tensorview.h"
// #include <iostream>

// namespace functor {

// template <typename Device, typename Index, typename IndexGrid, unsigned NDim>
// void PrintValuesFunctor<Device, Index, IndexGrid, NDim>::operator()() {
//     std::cout << "INSIDE SAMPLE FUNCTOR::PRINTVALUEFUNCTOR \n";
//     std::cout << "inputTensorView:\n";
//     // tv::printTensorView(indicesIn);
// }
// // template <typename Index, unsigned NDim>
// // struct PrintValuesFunctor<tv::CPU, Index, NDim>
// // {
// //     void operator()(const tv::CPU& d, tv::TensorView<const Index> input)
// //     {
// //         std::cout << "INSIDE FUNCTOR Input values:\n";
// //     }
// // };

// // template struct PrintValuesFunctor<tv::CPU, int64_t, int64_t, 3>;

// } // namespace functor
