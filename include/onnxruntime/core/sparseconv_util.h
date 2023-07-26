#pragma once

#include "onnxruntime_cxx_api.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include <vector>
#include <unordered_map>

class SparseConvTensor {
public:
    SparseConvTensor(Ort::Value& features_tensor, Ort::Value& indices_tensor,
                    const std::vector<int64_t>& spatial_shape, int64_t batch_size,
                    Ort::Value& grid_tensor)
    // SparseConvTensor(const Ort::Value& features_tensor,
    //              const Ort::Value& indices_tensor,
    //              const std::vector<int64_t>& spatial_shape,
    //              int64_t batch_size,
    //              const Ort::Value& grid_tensor)
    : features(std::move(features_tensor)),
      indices(std::move(indices_tensor)),
      spatial_shape(spatial_shape),
      batch_size(batch_size),
      grid(std::move(grid_tensor)) {}

    // Ort::Value dense(bool channels_first = true) {
    //     return dense_tensor;
    // }

    int64_t GetSpatialSize() {
        int64_t size = 1;
        for (const auto& dim : spatial_shape) {
            size *= dim;
        }
        return size;
    }

    // Ort::Value find_indice_pair(const KeyType& key) {
    //     auto it = indice_dict.find(key);
    //     if (it != indice_dict.end()) {
    //         return it->second;
    //     }
    //     return nullptr;
    // }

    Ort::Value features;
    Ort::Value indices;
    std::vector<int64_t> spatial_shape;
    int64_t batch_size;
    // std::unordered_map<nullptr, Ort::Value> indice_dict;
    Ort::Value grid;
};
    