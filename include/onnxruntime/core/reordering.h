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

#ifndef SPARSE_GATHER_FUNCTOR_H_
#define SPARSE_GATHER_FUNCTOR_H_
// #include <spconv/reordering.h>

namespace functor {

template <typename Device, typename T, typename Index>
struct SparseGatherFunctor {
  void operator()(/*const tv::CPU& d,*/ tv::TensorView<T> buffer, tv::TensorView<const T> features,
                  tv::TensorView<const Index> indices, int size) {
    std::cout << "INSIDE THE GATHER FUNCTOR\n\nPRINTING THE TENSORVIEW VALUES:\n\n";
    int numPlanes = features.dim(1);
    // std::cout << "numPlanes:" << numPlanes << "\n";
    // std::cout << "sizeof(T) * numPlanes:" << sizeof(T) * numPlanes << "\n";
    for (int i = 0; i < size; ++i) {
      // std::cout << "buffer.data() INSIDE Loop:" << buffer.data() << "\n\n";
      // std::cout << "i and buffer.data() + i * numPlanes:" << i << "       " << buffer.data() + i * numPlanes << "\n\n";
      std::memcpy(buffer.data() + i * numPlanes,
                  features.data() + indices[i] * numPlanes,
                  sizeof(T) * numPlanes);
    }
  }
};

template <typename Device, typename T, typename Index>
struct SparseScatterAddFunctor {
  // void operator()(/*const tv::CPU& d,*/ tv::TensorView<T> outFeatures,
  //                 tv::TensorView<const T> buffer, tv::TensorView<const Index> indices,
  //                 int size /*, bool stable*/) {
  void operator()(/*const tv::CPU& d,*/ tv::TensorView<T> outFeatures,
                  tv::TensorView<const T> buffer, tv::TensorView<const Index> indices,
                  int size /*, bool stable*/) {
    int numPlanes = outFeatures.dim(1);
    const T* buf = buffer.data();
    T* out = outFeatures.data();
    for (int i = 0; i < size; ++i) {
      buf = buffer.data() + i * numPlanes;
      out = outFeatures.data() + indices[i] * numPlanes;
      for (int j = 0; j < numPlanes; ++j){
        out[j] += buf[j];
      }
    }
  }
};

} // namespace functor
#endif
