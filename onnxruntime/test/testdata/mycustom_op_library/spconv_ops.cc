#include "spconv_ops.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <tuple>
#include <random>

#include "core/tensorview/tensorview.h"
#include "core/common/common.h"
#include "core/common/narrow.h"
#include <gsl/gsl>
#include "core/indice.h"
#include "core/ort_utils.h"
#include "core/sparseconv_util.h"
#include "core/reordering.h"

#include "core/common/safeint.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/op_kernel_type_control.h"

#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);
#endif

static const char* c_OpDomain = "test.my.customop";
constexpr int NDim = 3;

std::vector<int64_t> get_conv_output_size(const int64_t* input_size,
                                          const int64_t* kernel_size,
                                          const int64_t* stride,
                                          const int64_t* padding,
                                          const int64_t* dilation) {
    std::vector<int64_t> output_size;
    output_size.reserve(NDim);
    for (int64_t i = 0; i < NDim; ++i) {
        // int64_t size = (input_size[i] + 2 * padding[i] - dilation[i] *
        //     (kernel_size[i] - 1) - 1) / stride[i] + 1;
        int64_t size = std::floor((input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1;
        if (kernel_size[i] == -1) {
            output_size.push_back(1);
        } else {
            output_size.push_back(size);
        }
    }
    return output_size;
}

class GetIndicesPairOperator {
public:
    GetIndicesPairOperator(const OrtKernelInfo* k_info) : info_copy_(nullptr), op_slice_(nullptr) {
        Ort::ConstKernelInfo info{k_info};
        info_copy_ = info.Copy();
        const char* slice_type_constraint_names[2] = {"T", "Tind"};
        ONNXTensorElementDataType slice_type_constraint_values[2] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
        op_slice_ = Ort::Op::Create(info_copy_, "Slice", "", 13, slice_type_constraint_names, slice_type_constraint_values, 2, nullptr, 0, 4, 1);
    }
    // std::tuple<Ort::Value, Ort::Value, Ort::Value>
    /*std::vector<Ort::Value>*/std::tuple<int64_t, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> getIndicesPair(OrtKernelContext* context ) {
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::KernelContext ctx(context);
    auto indices = ctx.GetInput(1);
    auto batchSize = ctx.GetInput(2);
    auto spatialShape = ctx.GetInput(3);
    auto kernelSize = ctx.GetInput(6);
    auto stride = ctx.GetInput(7);
    auto padding = ctx.GetInput(8);
    auto dilation = ctx.GetInput(9);
    auto subM = ctx.GetInput(11);
    auto transpose = ctx.GetInput(12);

    //Getdata
    const auto* indicesData = indices.GetTensorData<int64_t>();
    const int64_t batchSizeData = *(batchSize.GetTensorData<int64_t>());
    const auto* spatialShapeData = spatialShape.GetTensorData<int64_t>();
    const auto* kernelSizeData = kernelSize.GetTensorData<int64_t>();
    const auto* strideData = stride.GetTensorData<int64_t>();
    const auto* paddingData = padding.GetTensorData<int64_t>();
    const auto* dilationData = dilation.GetTensorData<int64_t>();
    const int64_t subMData = *(subM.GetTensorData<int64_t>());
    const int64_t transposeData = *(transpose.GetTensorData<int64_t>());

    const auto indicesShape = indices.GetTensorTypeAndShapeInfo().GetShape();

    bool subm_bool = subMData != 0;
    bool transpose_bool = transposeData != 0;

    std::cout << "indicesShape:" << indicesShape[0] << "      " << indicesShape[1] <<  "\n";
    // Print the indices
    std::cout << "indicesData From Get_indices_pair: ";
    for (int64_t i = 0; i < indicesShape[0]; ++i) {
        for (int64_t j = 0; j < indicesShape[1]; ++j) {
            const int64_t index = indicesData[i * indicesShape[1] + j];
            std::cout << index << " ";
        }
        std::cout << std::endl;
    }

    auto numAct = indicesShape[0];
    int64_t coorDim = indicesShape[1] - 1; // batchIdx + xyz

    std::cout << "subm_bool: " << subm_bool << std::endl;
    std::cout << "transpose_bool: " << transpose_bool << std::endl;
    std::cout << "\nnumAct: " << numAct << std::endl;
    std::cout << "coorDim: " << coorDim << std::endl;

    std::vector<int64_t> out_spatial_size = get_conv_output_size(spatialShapeData, kernelSizeData, strideData, paddingData, dilationData);
    std::cout << "out_spatial_shape from getIndicesPair:\n\n";
    for (const auto& size : out_spatial_size) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
    std::vector<int64_t> outSpatialShapeVec = {static_cast<int64_t>(out_spatial_size.size())};
    Ort::Value outSpatialShape = Ort::Value::CreateTensor<int64_t>(mem_info, out_spatial_size.data(), out_spatial_size.size(), outSpatialShapeVec.data(), outSpatialShapeVec.size());


    const size_t kernel_elem_count = kernelSize.GetTensorTypeAndShapeInfo().GetElementCount();
    const size_t outSpatialShape_elem_count = outSpatialShape.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto* outSpatialShapeData = outSpatialShape.GetTensorData<int64_t>();


    auto kernelVolume = kernelSizeData[0];
    for (size_t i = 1; i < kernel_elem_count; ++i) {
      kernelVolume *= kernelSizeData[i];
    } // kernelVolume = 3 * 3 *3 = 27 -> [27]
    std::cout << "Updated KernelVol: " << kernelVolume << std::endl;

    auto outputVolume = outSpatialShapeData[0];
    for (size_t i = 1; i < outSpatialShape_elem_count; ++i) {
      std::cout << "outSpatialShapeData:" << outSpatialShapeData[i] << "\n";
      outputVolume *= outSpatialShapeData[i];
    } // OutputVolume = 5 * 10 *10 = 500
    std::cout << "Updated outputVolume: " << outputVolume << std::endl;


    // torch::Tensor indiceNum = torch::zeros( {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
    //IndicesNumTensor
    int64_t fill_value = 0;
    std::vector<int64_t> indiceNumVector(kernelVolume);
    std::fill(indiceNumVector.begin(), indiceNumVector.end(), fill_value);

    std::vector<int64_t> IndiceNumShape = {kernelVolume};
    // std::copy(indiceNumVector.begin(), indiceNumVector.end(), IndiceNumValues.begin());
    Ort::Value outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indiceNumVector.data(), kernelVolume, IndiceNumShape.data(), IndiceNumShape.size());

    // int64_t* outputData = outputTensor.GetTensorMutableData<int64_t>();
    // std::copy(indiceNumVector.begin(), indiceNumVector.end(), outputData); // check if this works

    int64_t indice_pair_fill_size = kernelVolume * 2 * numAct;
    int64_t indice_pair_fill_value = -1;
    std::vector<int64_t> indicePairVector(indice_pair_fill_size);
    std::fill(indicePairVector.begin(), indicePairVector.end(), indice_pair_fill_value);

    std::cout << "\n";
    std::cout << "Size of indicePairVector: " << indicePairVector.size() << "\n";

    std::vector<int64_t> IndicePairShape = {kernelVolume, 2, numAct};
    Ort::Value indice_pair_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indicePairVector.data(), indice_pair_fill_size, IndicePairShape.data(), IndicePairShape.size());

    int64_t* indice_pair_outputData = indice_pair_outputTensor.GetTensorMutableData<int64_t>();
    // std::copy(indicePairVector.begin(), indicePairVector.end(), indice_pair_outputData);

    std::cout << "indicePair outputData: ";
    for (int64_t i = 0; i < indice_pair_fill_size; ++i) {
        std::cout << indice_pair_outputData[i] << " ";
    }
    std::cout << "\n";

    //  torch::Tensor gridOut = torch::full({batchSize * outputVolume}, -1, torch::dtype(torch::kInt32).device(indices.device()));
    // gridOutTensor
    int64_t gridOut_fill_size = batchSizeData * outputVolume;
    int64_t gridOut_fill_value = -1;
    std::vector<int64_t> gridOutVector(gridOut_fill_size);
    std::fill(gridOutVector.begin(), gridOutVector.end(), gridOut_fill_value);

    std::vector<int64_t> gridOutShape = {batchSizeData * outputVolume};
    Ort::Value gridOut_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, gridOutVector.data(), gridOut_fill_size, gridOutShape.data(), gridOutShape.size());

    int64_t* gridOut_outputData = gridOut_outputTensor.GetTensorMutableData<int64_t>();
    // std::copy(gridOutVector.begin(), gridOutVector.end(), gridOut_outputData);

    std::cout << "gridOut outputData: ";
    for (int64_t i = 0; i < gridOut_fill_size; ++i) {
        std::cout << gridOut_outputData[i] << " ";
    }
    std::cout << "\n";

    int64_t numActOut = -1;
    tv::SimpleVector<int64_t, NDim> outSpatialShape32;
    tv::SimpleVector<int64_t, NDim> kernelSize32;
    tv::SimpleVector<int64_t, NDim> stride32;
    tv::SimpleVector<int64_t, NDim> padding32;
    tv::SimpleVector<int64_t, NDim> dilation32;

    // auto indicePairUnique =
    // torch::full({indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
    //             torch::dtype(torch::kInt32).device(indices.device()));
    // int64_t indicePairUnique_fill_size = indice_pair_fill_size / 2 + 1;
    // int64_t indicePairUnique_fill_value = std::numeric_limits<int>::max();
    // std::vector<int64_t> indicePairUniqueVector(indicePairUnique_fill_size);
    // std::fill(indicePairUniqueVector.begin(), indicePairUniqueVector.end(), indicePairUnique_fill_value);

    // std::vector<int64_t> indicePairUniqueOutShape = {indicePairUnique_fill_size};
    // Ort::Value indicePairUnique_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indicePairUniqueVector.data(), indicePairUnique_fill_size, indicePairUniqueOutShape.data(), indicePairUniqueOutShape.size());

    // int64_t* indicePairUnique_outputData = indicePairUnique_outputTensor.GetTensorMutableData<int64_t>();
    // std::copy(indicePairUniqueVector.begin(), indicePairUniqueVector.end(), indicePairUnique_outputData);


    // UPDATE CONV ATTRIBUTE
    for (int64_t i = 0; i < 3 /*NDIM*/; ++i) {
      outSpatialShape32.push_back(outSpatialShapeData[i]);
      kernelSize32.push_back(kernelSizeData[i]);
      if (subMData) {
        stride32.push_back(1);
        padding32.push_back(kernelSizeData[i] / 2);
        dilation32.push_back(dilationData[i]);
      } else {
        stride32.push_back(strideData[i]);
        padding32.push_back(paddingData[i]);
        dilation32.push_back(dilationData[i]);
      }
    }

    int64_t outInds_fill_size = (numAct * kernelVolume) * (coorDim + 1);
    int64_t outInds_fill_value = 0;
    std::vector<int64_t> outIndsVector(outInds_fill_size);
    std::fill(outIndsVector.begin(), outIndsVector.end(), outInds_fill_value);

    std::vector<int64_t> outIndsShape = {numAct * kernelVolume, coorDim + 1};
    Ort::Value outInds_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, outIndsVector.data(), outInds_fill_size, outIndsShape.data(), outIndsShape.size());
    std::cout << "After outInds_outputTensor creation\n";

    int64_t* outInds_outputData = outInds_outputTensor.GetTensorMutableData<int64_t>();

    std::cout << "outInds outputData: ";
    for (int64_t i = 0; i < outInds_fill_size; ++i) {
        std::cout << outInds_outputData[i] << " ";
    }
    std::cout << "\n";

    auto getIndicesPairFtor = functor::CreateConvIndicePairFunctor<tv::CPU, int64_t, int64_t, NDim>();

    std::cout << "Indices retrieved\n";
    auto indicesTensorView = tv::ort2tv<const int64_t>(indicesData, indicesShape);
    auto outIdsTensorView = tv::ort2tv<int64_t>(outInds_outputTensor);
    auto gridOutTensorView = tv::ort2tv<int64_t>(gridOut_outputTensor);
    auto indice_pair_outputTensorView = tv::ort2tv<int64_t>(indice_pair_outputTensor);
    auto outputTensorView = tv::ort2tv<int64_t>(outputTensor);

    numActOut = getIndicesPairFtor(
        indicesTensorView,
        outIdsTensorView, gridOutTensorView,
        indice_pair_outputTensorView, outputTensorView, kernelSize32,
        stride32, padding32, dilation32, outSpatialShape32);

    std::cout << "PRINT Tensorview Values in getIndicesPair function \n";
    // Print TensorView Values
    std::cout << "indicesTensorView:\n";
    tv::printTensorView(indicesTensorView);

    std::cout << "outIdsTensorView:\n";
    tv::printTensorView(outIdsTensorView);

    std::cout << "gridOutTensorView:\n";
    tv::printTensorView(gridOutTensorView);

    std::cout << "indice_pair_outputTensorView:\n";
    tv::printTensorView(indice_pair_outputTensorView);

    std::cout << "outputTensorView:\n";
    tv::printTensorView(outputTensorView);

    std::cout << "numActOut:" << numActOut << "\n";

    // BEGIN OF SLICE
    std::vector<int64_t> sliceOutShape = {numActOut, 4};
    std::vector<int64_t> SliceOutValues(numActOut * 4); //0
    auto outputSliceTensor = Ort::Value::CreateTensor<int64_t>(mem_info, SliceOutValues.data(), SliceOutValues.size(), sliceOutShape.data(), sliceOutShape.size());

    int64_t* outputSliceData = outputSliceTensor.GetTensorMutableData<int64_t>();
    std::cout << "outputSliceData before:" << outputSliceData[0] << "\n";

    std::vector<int64_t> sliceStartData = {0};
    std::vector<int64_t> raw_start_shape = {1};
    auto startValueOrt = Ort::Value::CreateTensor(mem_info, sliceStartData.data(), sliceStartData.size(), raw_start_shape.data(), raw_start_shape.size());

    std::vector<int64_t> sliceEndData = {numActOut};
    std::vector<int64_t> raw_end_value_shape = {1};
    auto endValueOrt = Ort::Value::CreateTensor(mem_info, sliceEndData.data(), sliceEndData.size(), raw_end_value_shape.data(), raw_end_value_shape.size());

    std::vector<int64_t> sliceAxesData = {0};
    std::vector<int64_t> raw_axes_value_shape = {1};
    auto axesValueOrt = Ort::Value::CreateTensor(mem_info, sliceAxesData.data(), sliceAxesData.size(), raw_axes_value_shape.data(), raw_axes_value_shape.size());
    std::cout << "SLICE TENSOR CREATION SUCCESSFUL\n";

    const Ort::Value slice_inputs[4] = {std::move(outInds_outputTensor), std::move(startValueOrt), std::move(endValueOrt), std::move(axesValueOrt)};

    Ort::Value slice_outputs[1] = {std::move(outputSliceTensor)};
    std::cout << "INVOKE SLICE\n\n";

    op_slice_.Invoke(context, slice_inputs, 4, slice_outputs, 1);

    int64_t slice_fill_size = numActOut * 4;

    std::cout << "slice_outputData outputData: ";
    int64_t* slice_updated = slice_outputs[0].GetTensorMutableData<int64_t>();
    for (int64_t i = 0; i < slice_fill_size; ++i) {
        std::cout << slice_updated[i] << " ";
    }
    std::cout << "\n";

    std::cout << "OutputSliceShape: [";
    for (size_t i = 0; i < sliceOutShape.size(); ++i) {
      std::cout << sliceOutShape[i];
      if (i != sliceOutShape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";

    std::cout << "indicesPairData Updated from GIP OP: ";
    int64_t* indice_pair_updated = indice_pair_outputTensor.GetTensorMutableData<int64_t>();
    std::cout << "indice_pair_fill_size from GIP OP: " << indice_pair_fill_size << "\n";
    for (int64_t i = 0; i < indice_pair_fill_size; ++i) {
        std::cout << indice_pair_updated[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "IndicesNum tensor shape: ";
    int64_t* indice_num_updated = outputTensor.GetTensorMutableData<int64_t>();
    for (size_t i = 0; i < IndiceNumShape.size(); ++i) {
        std::cout << IndiceNumShape[i] << " ";
    }
    std::cout << std::endl;

    // Print values
    std::cout << "IndicesNum tensor values: ";
    for (int64_t i = 0; i < kernelVolume; ++i) {
      std::cout << indice_num_updated[i] << " ";
    }
    std::cout << std::endl;

    // return the outputSliceTensor, indice_pair_outputTensor, outputTensor
    // std::vector<Ort::Value> outputs;
    // ort_inputs.push_back(std::move(slice_outputs[0]));
    // ort_inputs.push_back(std::move(indice_pair_outputTensor));
    // ort_inputs.push_back(std::move(outputTensor));

    // std::vector<Ort::Value> outputs;
    // outputs.push_back(std::move(outputSliceTensor));
    // outputs.push_back(std::move(indice_pair_outputTensor));
    // outputs.push_back(std::move(outputTensor));

    // ort_inputs.emplace_back(std::move(slice_outputs[0]));
    // ort_inputs.emplace_back(std::move(indice_pair_outputTensor));
    // ort_inputs.emplace_back(std::move(outputTensor));
    std::vector<int64_t> updatedData1(slice_updated, slice_updated + SliceOutValues.size());
    std::vector<int64_t> updatedData2(indice_pair_updated, indice_pair_updated + indice_pair_fill_size);

    std::vector<int64_t> updatedData3(indice_num_updated, indice_num_updated + kernelVolume);
    return std::make_tuple(numActOut, updatedData1, sliceOutShape, updatedData2, IndicePairShape, updatedData3, IndiceNumShape);
    // return std::make_tuple(std::move(slice_outputs[0]), std::move(indice_pair_outputTensor), std::move(outputTensor));
    // return std::tie(slice_outputs[0], indice_pair_outputTensor, outputTensor);  }
}
private:
    Ort::KernelInfo info_copy_;
    Ort::Op op_slice_;
};

struct IndicesPairKernel {
  IndicesPairKernel(const OrtKernelInfo* info);
  ~IndicesPairKernel();

  void Compute(OrtKernelContext* context) {
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    std::cout << "INSIDE GetIndicesPair Kernel IMPLEMENTATION\n";
    Ort::KernelContext ctx(context);
    auto indices = ctx.GetInput(1);
    auto batchSize = ctx.GetInput(2);
    // auto outSpatialShape = ctx.GetInput(2);
    auto spatialShape = ctx.GetInput(3);
    auto kernelSize = ctx.GetInput(6);
    auto stride = ctx.GetInput(7);
    auto padding = ctx.GetInput(8);
    auto dilation = ctx.GetInput(9);
    auto outPadding = ctx.GetInput(10);
    auto subM = ctx.GetInput(11);
    auto transpose = ctx.GetInput(12);

    //Getdata
    const auto* indicesData = indices.GetTensorData<int64_t>();
    const int64_t batchSizeData = *(batchSize.GetTensorData<int64_t>());
    // const int64_t* batchSizeData = batchSize.GetTensorData<int64_t>();
    // const auto* outSpatialShapeData = outSpatialShape.GetTensorData<int64_t>();
    const auto* spatialShapeData = spatialShape.GetTensorData<int64_t>();
    const auto* kernelSizeData = kernelSize.GetTensorData<int64_t>();
    const auto* strideData = stride.GetTensorData<int64_t>();
    const auto* paddingData = padding.GetTensorData<int64_t>();
    const auto* dilationData = dilation.GetTensorData<int64_t>();
    // const auto* outPaddingData = outPadding.GetTensorData<int64_t>();
    const int64_t subMData = *(subM.GetTensorData<int64_t>());
    const int64_t transposeData = *(transpose.GetTensorData<int64_t>());

    //Print the values
    std::cout << "PRINTING INPUTS FROM GETINDICESPAIR\n";
    // std::cout << "batchSizeData: " << batchSizeData << std::endl;
    // std::cout << "outSpatialShapeData: " << *outSpatialShapeData << std::endl;
    std::cout << "\n\n";

    // Get input shapes
    const auto indicesShape = indices.GetTensorTypeAndShapeInfo().GetShape();
    // const auto batchSizeShape = batchSize.GetTensorTypeAndShapeInfo().GetShape();
    // const auto outSpatialShapeShape = outSpatialShape.GetTensorTypeAndShapeInfo().GetShape();
    const auto spatialShapeShape = spatialShape.GetTensorTypeAndShapeInfo().GetShape();
    const auto kernelSizeShape = kernelSize.GetTensorTypeAndShapeInfo().GetShape();
    const auto strideShape = stride.GetTensorTypeAndShapeInfo().GetShape();
    const auto paddingShape = padding.GetTensorTypeAndShapeInfo().GetShape();
    const auto dilationShape = dilation.GetTensorTypeAndShapeInfo().GetShape();
    const auto outPaddingShape = outPadding.GetTensorTypeAndShapeInfo().GetShape();
    // const auto subMShape = subM.GetTensorTypeAndShapeInfo().GetShape();
    const auto transposeShape = transpose.GetTensorTypeAndShapeInfo().GetShape();

    bool subm_bool = subMData != 0;
    bool transpose_bool = transposeData != 0;

    std::cout << "PRINTING INDICES FROM GETINDICESPAIROP\n\n";
    std::cout << "indicesShape:" << indicesShape[0] << "      " << indicesShape[1] <<  "\n";
    // Print the indices
    std::cout << "indicesData From Get_indices_pair: ";
    for (int64_t i = 0; i < indicesShape[0]; ++i) {
        for (int64_t j = 0; j < indicesShape[1]; ++j) {
            const int64_t index = indicesData[i * indicesShape[1] + j];
            std::cout << index << " ";
        }
        std::cout << std::endl;
    }

    auto numAct = indicesShape[0];
    int64_t coorDim = indicesShape[1] - 1; // batchIdx + xyz

    std::cout << "subm_bool: " << subm_bool << std::endl;
    std::cout << "transpose_bool: " << transpose_bool << std::endl;
    std::cout << "\nnumAct: " << numAct << std::endl;
    std::cout << "coorDim: " << coorDim << std::endl;

    std::vector<int64_t> out_spatial_size = get_conv_output_size(spatialShapeData, kernelSizeData, strideData, paddingData, dilationData);
    std::cout << "out_spatial_shape from getIndicesPair:\n\n";
    for (const auto& size : out_spatial_size) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
    std::vector<int64_t> outSpatialShapeVec = {static_cast<int64_t>(out_spatial_size.size())};
    Ort::Value outSpatialShape = Ort::Value::CreateTensor<int64_t>(mem_info, out_spatial_size.data(), out_spatial_size.size(), outSpatialShapeVec.data(), outSpatialShapeVec.size());


    const size_t kernel_elem_count = kernelSize.GetTensorTypeAndShapeInfo().GetElementCount();
    const size_t outSpatialShape_elem_count = outSpatialShape.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto* outSpatialShapeData = outSpatialShape.GetTensorData<int64_t>();


    auto kernelVolume = kernelSizeData[0];
    for (size_t i = 1; i < kernel_elem_count; ++i) {
      kernelVolume *= kernelSizeData[i];
    } // kernelVolume = 3 * 3 *3 = 27 -> [27]
    // [-1, -1, -1, -1, -1....,]
    std::cout << "KernelVol Updated: " << kernelVolume << std::endl;

    auto outputVolume = outSpatialShapeData[0];
    for (size_t i = 1; i < outSpatialShape_elem_count; ++i) {
      std::cout << "outSpatialShapeData[i]:" << outSpatialShapeData[i] << "\n";
      outputVolume *= outSpatialShapeData[i];
    } // OutputVolume = 5 * 10 *10 = 500
    std::cout << "outputVolume Updated: " << outputVolume << std::endl;

    //MAIN LOGIC

    // torch::Tensor indiceNum = torch::zeros( {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
    //IndicesNumTensor
    int64_t fill_size = kernelVolume;
    int64_t fill_value = 0;
    std::vector<int64_t> indiceNumVector(fill_size);
    std::fill(indiceNumVector.begin(), indiceNumVector.end(), fill_value);

    std::cout << "indiceNumVector: ";
    for (const auto& value : indiceNumVector) {
        std::cout << value << " ";
    }
    // std::cout << "\n";
    // std::cout << "Size of indiceNumVector: " << indiceNumVector.size() << "\n";
    // std::cout << "After fill\n";

    std::vector<int64_t> IndiceNumShape = {kernelVolume};
    // std::copy(indiceNumVector.begin(), indiceNumVector.end(), IndiceNumValues.begin());
    Ort::Value outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indiceNumVector.data(), fill_size, IndiceNumShape.data(), IndiceNumShape.size());

    int64_t* outputData = outputTensor.GetTensorMutableData<int64_t>();
    std::copy(indiceNumVector.begin(), indiceNumVector.end(), outputData);

    // std::cout << "indiceNum outputData: ";
    // for (int64_t i = 0; i < fill_size; ++i) {
    //     std::cout << outputData[i] << " ";
    // }
    // std::cout << "\n";

    const auto IndiceNumOutShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "IndiceNumOutShape: (";
    for (size_t i = 0; i < IndiceNumOutShape.size(); ++i) {
        std::cout << IndiceNumOutShape[i];
        if (i != IndiceNumOutShape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";

    // torch::Tensor indicePairs = torch::full({kernelVolume, 2, numAct}, -1, torch::dtype(torch::kInt32).device(indices.device()));
    //IndicePairsTensor
    int64_t indice_pair_fill_size = kernelVolume * 2 * numAct;
    int64_t indice_pair_fill_value = -1;
    std::vector<int64_t> indicePairVector(indice_pair_fill_size);
    std::fill(indicePairVector.begin(), indicePairVector.end(), indice_pair_fill_value);

    std::cout << "indicePairVector: ";
    for (const auto& value : indicePairVector) {
        std::cout << value << " ";
    }
    std::cout << "\n";
    std::cout << "Size of indicePairVector: " << indicePairVector.size() << "\n";
    std::cout << "After fill\n";

    std::vector<int64_t> IndicePairShape = {kernelVolume, 2, numAct};
    Ort::Value indice_pair_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indicePairVector.data(), indice_pair_fill_size, IndicePairShape.data(), IndicePairShape.size());

    int64_t* indice_pair_outputData = indice_pair_outputTensor.GetTensorMutableData<int64_t>();
    std::copy(indicePairVector.begin(), indicePairVector.end(), indice_pair_outputData);

    std::cout << "indicePair outputData: ";
    for (int64_t i = 0; i < indice_pair_fill_size; ++i) {
        std::cout << indice_pair_outputData[i] << " ";
    }
    std::cout << "\n";

    const auto IndicePairOutShape = indice_pair_outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "IndicePairOutShape: (";
    for (size_t i = 0; i < IndicePairOutShape.size(); ++i) {
        std::cout << IndicePairOutShape[i];
        if (i != IndicePairOutShape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";

    //  torch::Tensor gridOut = torch::full({batchSize * outputVolume}, -1, torch::dtype(torch::kInt32).device(indices.device()));
    // gridOutTensor
    int64_t gridOut_fill_size = batchSizeData * outputVolume;
    int64_t gridOut_fill_value = -1;
    std::vector<int64_t> gridOutVector(gridOut_fill_size);
    std::fill(gridOutVector.begin(), gridOutVector.end(), gridOut_fill_value);

    std::cout << "gridOutVector: ";
    for (const auto& value : gridOutVector) {
        std::cout << value << " ";
    }
    std::cout << "\n";
    std::cout << "Size of gridOutVector: " << gridOutVector.size() << "\n";
    std::cout << "After fill\n";

    std::vector<int64_t> gridOutShape = {batchSizeData * outputVolume};
    Ort::Value gridOut_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, gridOutVector.data(), gridOut_fill_size, gridOutShape.data(), gridOutShape.size());

    int64_t* gridOut_outputData = gridOut_outputTensor.GetTensorMutableData<int64_t>();
    std::copy(gridOutVector.begin(), gridOutVector.end(), gridOut_outputData);

    // std::cout << "gridOut outputData: ";
    // for (int64_t i = 0; i < gridOut_fill_size; ++i) {
    //     std::cout << gridOut_outputData[i] << " ";
    // }
    // std::cout << "\n";

    const auto gridOutShape2 = gridOut_outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "gridOutShape after: (";
    for (size_t i = 0; i < gridOutShape2.size(); ++i) {
        std::cout << gridOutShape2[i];
        if (i != gridOutShape2.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";

    int64_t numActOut = -1;
    tv::SimpleVector<int64_t, NDim> outSpatialShape32;
    tv::SimpleVector<int64_t, NDim> kernelSize32;
    tv::SimpleVector<int64_t, NDim> stride32;
    tv::SimpleVector<int64_t, NDim> padding32;
    tv::SimpleVector<int64_t, NDim> dilation32;

    // auto indicePairUnique =
    // torch::full({indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
    //             torch::dtype(torch::kInt32).device(indices.device()));
    int64_t indicePairUnique_fill_size = indice_pair_fill_size / 2 + 1;
    int64_t indicePairUnique_fill_value = std::numeric_limits<int>::max();
    std::vector<int64_t> indicePairUniqueVector(indicePairUnique_fill_size);
    std::fill(indicePairUniqueVector.begin(), indicePairUniqueVector.end(), indicePairUnique_fill_value);

    std::cout << "Size of indicePairUniqueVector: " << indicePairUniqueVector.size() << "\n";
    std::cout << "After fill\n";

    std::vector<int64_t> indicePairUniqueOutShape = {indicePairUnique_fill_size};
    Ort::Value indicePairUnique_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indicePairUniqueVector.data(), indicePairUnique_fill_size, indicePairUniqueOutShape.data(), indicePairUniqueOutShape.size());

    int64_t* indicePairUnique_outputData = indicePairUnique_outputTensor.GetTensorMutableData<int64_t>();
    std::copy(indicePairUniqueVector.begin(), indicePairUniqueVector.end(), indicePairUnique_outputData);

    // std::cout << "indicePairUnique outputData: ";
    // for (int64_t i = 0; i < indicePairUnique_fill_size; ++i) {
    //     std::cout << indicePairUnique_outputData[i] << " ";
    // }
    // std::cout << "\n";

    const auto indicePairUniqueShape2 = indicePairUnique_outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "indicePairUniqueShape2 after: (";
    for (size_t i = 0; i < indicePairUniqueShape2.size(); ++i) {
        std::cout << indicePairUniqueShape2[i];
        if (i != indicePairUniqueShape2.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";

    // UPDATE CONV ATTRIBUTE
    for (int64_t i = 0; i < 3 /*NDIM*/; ++i) {
      outSpatialShape32.push_back(outSpatialShapeData[i]);
      kernelSize32.push_back(kernelSizeData[i]);
      if (subMData) {
        stride32.push_back(1);
        padding32.push_back(kernelSizeData[i] / 2);
        dilation32.push_back(dilationData[i]);
      } else {
        stride32.push_back(strideData[i]);
        padding32.push_back(paddingData[i]);
        dilation32.push_back(dilationData[i]);
      }
    }

    // REMAINING
    // torch::Tensor outInds =
    //     torch::zeros({numAct * kernelVolume, coorDim + 1},
    //                 torch::dtype(torch::kInt32).device(indices.device()));
    std::cout << "numAct:" << numAct << "\n";
    std::cout << "kernelVolume:" << kernelVolume << "\n";
    std::cout << "coorDim:" << coorDim << "\n\n";

    int64_t outInds_fill_size = (numAct * kernelVolume) * (coorDim + 1);
    int64_t outInds_fill_value = 0;
    std::vector<int64_t> outIndsVector(outInds_fill_size);
    std::fill(outIndsVector.begin(), outIndsVector.end(), outInds_fill_value);

    std::cout << "Size of outIndsVector: " << outIndsVector.size() << "\n";
    std::cout << "After OUTIDS fill\n";

    std::vector<int64_t> outIndsShape = {numAct * kernelVolume, coorDim + 1};
    Ort::Value outInds_outputTensor = Ort::Value::CreateTensor<int64_t>(mem_info, outIndsVector.data(), outInds_fill_size, outIndsShape.data(), outIndsShape.size());
    std::cout << "After outInds_outputTensor creation\n";

    int64_t* outInds_outputData = outInds_outputTensor.GetTensorMutableData<int64_t>();
    std::copy(outIndsVector.begin(), outIndsVector.end(), outInds_outputData);
    std::cout << "After outInds_outputData copying \n";

    std::cout << "outInds outputData: ";
    for (int64_t i = 0; i < outInds_fill_size; ++i) {
        std::cout << outInds_outputData[i] << " ";
    }
    std::cout << "\n";

    const auto outIndsShape2 = outInds_outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "outIndsShape2 after: (";
    for (size_t i = 0; i < outIndsShape2.size(); ++i) {
        std::cout << outIndsShape2[i];
        if (i != outIndsShape2.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";


    std::cout << "outSpatialShape32 size: " << outSpatialShape32.size() << std::endl;
    std::cout << "outSpatialShape32 data: ";
    for (size_t i = 0; i < outSpatialShape32.size(); ++i) {
      std::cout << outSpatialShape32[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "kernelSize32 size: " << kernelSize32.size() << std::endl;
    std::cout << "kernelSize32 data: ";
    for (size_t i = 0; i < kernelSize32.size(); ++i) {
      std::cout << kernelSize32[i] << " ";
    }
    std::cout << std::endl;

    auto getIndicesPairFtor = functor::CreateConvIndicePairFunctor<tv::CPU, int64_t, int64_t, NDim>();

    std::cout << "Indices retrieved\n";
    auto indicesTensorView = tv::ort2tv<const int64_t>(indicesData, indicesShape);
    auto outIdsTensorView = tv::ort2tv<int64_t>(outInds_outputTensor);
    auto gridOutTensorView = tv::ort2tv<int64_t>(gridOut_outputTensor);
    auto indice_pair_outputTensorView = tv::ort2tv<int64_t>(indice_pair_outputTensor);
    auto outputTensorView = tv::ort2tv<int64_t>(outputTensor);

    numActOut = getIndicesPairFtor(
        indicesTensorView,
        outIdsTensorView, gridOutTensorView,
        indice_pair_outputTensorView, outputTensorView, kernelSize32,
        stride32, padding32, dilation32, outSpatialShape32);

    std::cout << "numActOut:" << numActOut << "\n";

    // std::cout << "indicePairs TensorView Updated:\n";
    // tv::printTensorView(indice_pair_outputTensorView);
    // std::cout << std::endl;

    // std::cout << "indiceNum TensorView Updated:\n";
    // tv::printTensorView(outputTensorView);
    // std::cout << std::endl;

    // PRINT INDICESNUM OUTPUT FINAL TENSOR AFTER UPDATION
    // int64_t* outputDataFinal = outputTensor.GetTensorMutableData<int64_t>();
    std::vector<int64_t> outputShapeFinal = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t numElements = 1;
    for (int64_t dim : outputShapeFinal) {
      numElements *= dim;
    }
    // Print the values
    // for (size_t i = 0; i < numElements; ++i) {
    //   std::cout << "outputTensorFinal[" << i << "]: " << outputDataFinal[i] << std::endl;
    // }

    // PRINT OUTIDS OUTPUT FINAL TENSOR AFTER UPDATION
    // int64_t* outIdsDataFinal = outInds_outputTensor.GetTensorMutableData<int64_t>();
    // std::vector<int64_t> outIdsShapeFinal = outInds_outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    // size_t OutIdsnumElements = 1;
    // for (int64_t dim : outIdsShapeFinal) {
    //   OutIdsnumElements *= dim;
    // }
    // Print the values
    // for (size_t i = 0; i < OutIdsnumElements; ++i) {
    //   std::cout << "outIdsDataFinal[" << i << "]: " << outIdsDataFinal[i] << std::endl;
    // }

    const auto outIndsShapeFinal = outInds_outputTensor.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "outIndsShapeFinal after: (";
    for (size_t i = 0; i < outIndsShapeFinal.size(); ++i) {
        std::cout << outIndsShapeFinal[i];
        if (i != outIndsShapeFinal.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";

    // //BEGIN OF SLICE
    std::vector<int64_t> sliceOutShape = {numActOut, 4};
    std::vector<int64_t> SliceOutValues(numActOut * 4);  // Create a vector to hold the tensor values
    auto outputSliceTensor = Ort::Value::CreateTensor<int64_t>(mem_info, SliceOutValues.data(), SliceOutValues.size(), sliceOutShape.data(), sliceOutShape.size());

    int64_t* outputSliceData = outputSliceTensor.GetTensorMutableData<int64_t>();
    std::cout << "outputSliceData before:" << outputSliceData[0] << "\n";

    std::vector<int64_t> sliceStartData = {0};
    std::vector<int64_t> raw_start_shape = {1};
    auto startValueOrt = Ort::Value::CreateTensor(mem_info, sliceStartData.data(), sliceStartData.size(), raw_start_shape.data(), raw_start_shape.size());

    std::vector<int64_t> sliceEndData = {numActOut};
    std::vector<int64_t> raw_end_value_shape = {1};
    auto endValueOrt = Ort::Value::CreateTensor(mem_info, sliceEndData.data(), sliceEndData.size(), raw_end_value_shape.data(), raw_end_value_shape.size());

    // int64_t* axesValueOrtData = axesValueOrt.GetTensorMutableData<int64_t>();
    // std::cout << "axesValueOrtData:" << axesValueOrtData[0] << "\n\n";

    std::vector<int64_t> sliceAxesData = {0};
    std::vector<int64_t> raw_axes_value_shape = {1};
    auto axesValueOrt = Ort::Value::CreateTensor(mem_info, sliceAxesData.data(), sliceAxesData.size(), raw_axes_value_shape.data(), raw_axes_value_shape.size());
    std::cout << "SLICE TENSOR CREATION SUCCESSFUL\n";

    // std::vector<int64_t> stepValueData = {1, numActOut};
    // int64_t stepValueData = 1;
    // std::vector<int64_t> stepValueShape = {1};
    // auto stepValueOrt = Ort::Value::CreateTensor<int64_t>(mem_info, &stepValueData, 1, stepValueShape.data(), stepValueShape.size());

    const Ort::Value slice_inputs[4] = {std::move(outInds_outputTensor), std::move(startValueOrt), std::move(endValueOrt), std::move(axesValueOrt)};

    Ort::Value slice_outputs[1] = {std::move(outputSliceTensor)};
    std::cout << "INVOKE SLICE\n\n";
    op_slice_.Invoke(context, slice_inputs, 4, slice_outputs, 1);

    int64_t slice_fill_size = numActOut * 4;
    // int64_t slice_fill_value = 0;
    std::vector<int64_t> sliceVector(slice_fill_size);

    int64_t* slice_outputData = slice_outputs[0].GetTensorMutableData<int64_t>();
    std::vector<int64_t> outputSliceShape = slice_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // std::copy(sliceVector.begin(), sliceVector.end(), slice_outputData);

    // std::cout << "Output Slice Data:\n";
    // for (int64_t i = 0; i < outputSliceShape[0]; i++) {
    //     for (int64_t j = 0; j < outputSliceShape[1]; j++) {
    //         std::cout << outputSliceData[i * outputSliceShape[1] + j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "slice_outputData outputData: ";
    // for (int64_t i = 0; i < slice_fill_size; ++i) {
    //     std::cout << slice_outputData[i] << " ";
    // }
    // std::cout << "\n";

    std::cout << "OutputSliceShape: [";
    for (size_t i = 0; i < sliceOutShape.size(); ++i) {
      std::cout << sliceOutShape[i];
      if (i != sliceOutShape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]\n";

    // END OF SLICE

    std::cout << "indicePair outputData Updated: ";
    for (int64_t i = 0; i < indice_pair_fill_size; ++i) {
        std::cout << indice_pair_outputData[i] << " ";
    }
    std::cout << "\n";

    int64_t* indicePairOutData = indice_pair_outputTensor.GetTensorMutableData<int64_t>();

    //Final Shape
    std::cout << "sliceOutputShape: ";
    for (const auto& shape : sliceOutShape) {
      std::cout << shape << " ";
    }
    std::cout << "\n";

    std::cout << "IndicePairShape: ";
    for (const auto& shape : IndicePairShape) {
      std::cout << shape << " ";
    }
    std::cout << "\n";

    std::cout << "IndiceNumShape: ";
    for (const auto& shape : IndiceNumShape) {
      std::cout << shape << " ";
    }
    std::cout << "\n";

    //setup Output Vectors
    std::vector<std::vector<int64_t>> output_shapes = {
      sliceOutShape,
      IndicePairShape,
      IndiceNumShape
    };

    const size_t num_outputs = output_shapes.size();
    std::cout << "num_outputs:" << num_outputs << "\n";
    // std::vector<Ort::Value*> output_tensors(num_outputs);
    // std::vector<int64_t*> output_data(num_outputs);

    std::vector<int64_t*> tensordata = {
      slice_outputData,
      indicePairOutData,
      outputData
    };

    for (size_t i = 0; i < num_outputs; i++) {
      auto output = ctx.GetOutput(i, output_shapes[i]);
      int64_t* out = output.GetTensorMutableData<int64_t>();
      const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
      std::cout << "tensor size:" << size << "\n";
      for (size_t j = 0; j < size; j++) {
        out[j] = tensordata[i][j];
      }
    }
  }
  private:
    Ort::KernelInfo info_copy_{nullptr};
    Ort::Op op_slice_{nullptr};
};

IndicesPairKernel::IndicesPairKernel(const OrtKernelInfo* k_info) {
    Ort::ConstKernelInfo info{k_info};
    info_copy_ = info.Copy();

    // const char* unsqueeze_type_constraint_names[1] = {"T"};
    // ONNXTensorElementDataType unsqueeze_type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
    // op_unsqueeze_ = Ort::Op::Create(info_copy_, "Unsqueeze", "", 13,
    //                           unsqueeze_type_constraint_names,
    //                           unsqueeze_type_constraint_values,
    //                           1, nullptr, 0, 2, 1);

    // constexpr int64_t axis_value = 0;
    // auto axis = Ort::OpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    // Ort::OpAttr attr_list[1] = {std::move(axis)};
    // op_concat_ = Ort::Op::Create(info_copy_, "Concat", "", 13,
    //                             unsqueeze_type_constraint_names,
    //                             unsqueeze_type_constraint_values,
    //                             1, attr_list, 1, 1, 1);

    const char* slice_type_constraint_names[2] = {"T", "Tind"};
    ONNXTensorElementDataType slice_type_constraint_values[2] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};

    // ONNX_NAMESPACE::TensorProto t_proto;
    // t_proto.set_data_type(TensorProto::INT32);
    // t_proto.mutable_dims()->Add(1);
    // t_proto.mutable_int32_data()->Add(0);

    // // constexpr int64_t value = 0;
    // Ort::OpAttr valueAttr("value", t_proto);
    // auto valueAttr = Ort::OpAttr("value", &value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    // Ort::OpAttr valAttrList[1] = {std::move(valueAttr)};
    // op_constant_of_shape_ = Ort::Op::Create(info_copy_, "ConstantOfShape", "", 9,
    //                             COS_type_constraint_names,
    //                             COS_type_constraint_values,
    //                             1, valAttrList, 1, 1, 1);

    op_slice_ = Ort::Op::Create(info_copy_, "Slice", "", 13,
                                slice_type_constraint_names,
                                slice_type_constraint_values,
                                2, nullptr, 0, 4, 1);
}

struct GetIndicesPair : Ort::CustomOpBase<GetIndicesPair, IndicesPairKernel> {
  // void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
  //   return std::make_unique<IndicesPairKernel>().release();
  // };
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new IndicesPairKernel(info); };

  const char* GetName() const { return "GetIndicesPair"; };

  size_t GetInputTypeCount() const { return 13; };

  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };

  // size_t GetOutputTypeCount() const { return 1; };
  size_t GetOutputTypeCount() const { return 3; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };
};

//START OF SPCONV
struct SparseConvKernel {
  SparseConvKernel(const OrtKernelInfo* k_info) : get_indices_pair_op_(k_info), indicesPairOperator(k_info) {
    Ort::ConstKernelInfo info{k_info};
    info_copy_ = info.Copy();

    const char* gemm_type_constraint_names[1] = {"T"};
    ONNXTensorElementDataType gemm_type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

    constexpr int64_t attr_value = 0;
    constexpr float alphaBeta = 1.0f;
    auto transA = Ort::OpAttr("transA", &attr_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    auto transB = Ort::OpAttr("transB", &attr_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    auto alpha = Ort::OpAttr("alpha", &alphaBeta, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);
    auto beta = Ort::OpAttr("beta", &alphaBeta, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);


    Ort::OpAttr attr_list[4] = {std::move(transA), std::move(transB), std::move(alpha), std::move(beta)};
    op_gemm_ = Ort::Op::Create(info_copy_, "Gemm", "", 13,
                                gemm_type_constraint_names,
                                gemm_type_constraint_values,
                                1, attr_list, 4, 2, 1);

    const char* cast_type_constraint_names[2] = {"T1", "T2"};
    ONNXTensorElementDataType cast_type_constraint_values[2] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

    constexpr int64_t attr_to = 1;
    auto op_attr_to = Ort::OpAttr("to", &attr_to, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    Ort::OpAttr cast_attr_list[1] = {std::move(op_attr_to)};

    op_cast_ = Ort::Op::Create(info_copy_, "Cast", "", 13,
                                cast_type_constraint_names,
                                cast_type_constraint_values,
                                2, cast_attr_list, 1, 1, 1);

  }
  ~SparseConvKernel() {}

  void Compute(OrtKernelContext* context) {
    // IMPLEMENTATION
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    std::cout << "INSIDE SparseConvolution Kernel IMPLEMENTATION\n";
    Ort::KernelContext ctx(context);
    auto features = ctx.GetInput(0);
    auto indices_th = ctx.GetInput(1);
    auto batchSize = ctx.GetInput(2);
    auto spatialShape = ctx.GetInput(3);

    auto inChannels = ctx.GetInput(4);
    auto outChannels = ctx.GetInput(5);
    auto kernelSize  = ctx.GetInput(6);
    auto stride = ctx.GetInput(7);
    auto padding = ctx.GetInput(8);
    auto dilation = ctx.GetInput(9);

    auto outPadding = ctx.GetInput(10);
    auto subM = ctx.GetInput(11);
    auto transpose = ctx.GetInput(12);
    auto grid = ctx.GetInput(13);

    //Getdata
    const auto* featuresData = features.GetTensorData<int64_t>();
    const auto* indicesData = indices_th.GetTensorData<int64_t>();
    const int64_t batchSizeData = *(batchSize.GetTensorData<int64_t>());
    const auto* spatialShapeData = spatialShape.GetTensorData<int64_t>();

    const int64_t inChannelsData = *(inChannels.GetTensorData<int64_t>());
    const int64_t outChannelsData = *(outChannels.GetTensorData<int64_t>());
    const auto* kernelSizeData = kernelSize.GetTensorData<int64_t>();
    const auto* strideData = stride.GetTensorData<int64_t>();
    const auto* paddingData = padding.GetTensorData<int64_t>();
    const auto* dilationData = dilation.GetTensorData<int64_t>();
    const int64_t subMData = *(subM.GetTensorData<int64_t>());
    const int64_t transposeData = *(transpose.GetTensorData<int64_t>());
    const auto* gridData = grid.GetTensorData<int64_t>();

    // Get input shapes
    const auto featuresShape = features.GetTensorTypeAndShapeInfo().GetShape();
    const auto indicesShape = indices_th.GetTensorTypeAndShapeInfo().GetShape();
    const auto spatialShapeShape = spatialShape.GetTensorTypeAndShapeInfo().GetShape();

    // const auto inChannelShape = inChannels.GetTensorTypeAndShapeInfo().GetShape();
    // const auto outChannelShape = outChannels.GetTensorTypeAndShapeInfo().GetShape();
    const auto kernelSizeShape = kernelSize.GetTensorTypeAndShapeInfo().GetShape();
    const auto strideShape = stride.GetTensorTypeAndShapeInfo().GetShape();
    const auto paddingShape = padding.GetTensorTypeAndShapeInfo().GetShape();
    const auto dilationShape = dilation.GetTensorTypeAndShapeInfo().GetShape();
    const auto outPaddingShape = outPadding.GetTensorTypeAndShapeInfo().GetShape();
    // const auto subMShape = subM.GetTensorTypeAndShapeInfo().GetShape();
    const auto transposeShape = transpose.GetTensorTypeAndShapeInfo().GetShape();
    const auto gridShape = grid.GetTensorTypeAndShapeInfo().GetShape();

    bool subm_bool = subMData != 0;
    bool transpose_bool = transposeData != 0;
    bool inverse_bool = false;


    std::cout << "subm_bool: " << subm_bool << std::endl;
    std::cout << "transpose_bool: " << transpose_bool << std::endl;
    std::cout << "inverse_bool:" << inverse_bool << std::endl;

    // Print the values of indicesData
    std::cout << "indicesData: ";
    for (size_t i = 0; i < indicesShape.size(); ++i) {
        std::cout << indicesData[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "spatialShapeData: ";
    std::cout << spatialShapeData << " ";
    std::cout << std::endl;

    size_t feature_num_elements = 1;
    for (const auto& dim : featuresShape) {
      feature_num_elements *= dim;
    }

    size_t grid_num_elements = 1;
    for (const auto& dim : gridShape) {
      grid_num_elements *= dim;
    }

    auto features_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(featuresData), sizeof(int64_t) * feature_num_elements, featuresShape.data(), featuresShape.size());

    std::vector<int64_t> spatial_shape(spatialShapeData, spatialShapeData + spatialShapeShape[0]);

    Ort::Value indices_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(indicesData), sizeof(int64_t) * indicesShape[0] * indicesShape[1], indicesShape.data(), indicesShape.size());
    //Create GridTensor int64_t grid_data[] = {1, 2, 3, 4};

    Ort::Value grid_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(gridData), sizeof(int64_t) * grid_num_elements,
                                                        gridShape.data(), gridShape.size());

    std::cout << "features Created: ";
    const int64_t* features_updated = features_tensor.GetTensorData<int64_t>();
    const auto featureShape2 = features_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const size_t feature_fill_size = features_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    const auto finput_rank = features_tensor.GetTensorTypeAndShapeInfo().GetDimensionsCount();
    std::cout << "feature_fill_size: " << feature_fill_size << "\n";
    std::cout << "feature_tensor input rank:" << finput_rank << "\n";
    for (size_t i = 0; i < feature_fill_size; ++i) {
        std::cout << features_updated[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Indicespair tensor shape: ";
    for (size_t i = 0; i < featureShape2.size(); ++i) {
        std::cout << featureShape2[i] << " ";
    }
    std::cout << std::endl;

    SparseConvTensor spconv_tensor(features_tensor, indices_tensor, spatial_shape, batchSizeData, grid_tensor);

    const Ort::Value& indices1 = spconv_tensor.indices;
    std::vector<int64_t> indices1_shape = indices1.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "indices1_shape:" << indices1_shape[0] << "      " << indices1_shape[1] <<  "\n";
    const int64_t* indices1_data = indices1.GetTensorData<int64_t>();

    // Print the indices
    std::cout << "indicesData From SpconvTensor: ";
    for (int64_t i = 0; i < indices1_shape[0]; ++i) {
        for (int64_t j = 0; j < indices1_shape[1]; ++j) {
            const int64_t index = indices1_data[i * indices1_shape[1] + j];
            std::cout << index << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Succesfully created SparseConvTensor\n";

    //Define Weight
    // self.weight = Parameter(torch.Tensor(*kernel_size, in_channels, out_channels))
    std::cout << "weight creation/ outChannelsData:" << outChannelsData << "\n";

    std::vector<int64_t> weightShape = {kernelSizeData[0], kernelSizeData[1], kernelSizeData[2], inChannelsData, outChannelsData};
    // std::vector<int64_t> weightData(weightShape.size());
    size_t weight_elem_count = kernelSizeData[0] * kernelSizeData[1] * kernelSizeData[2] * inChannelsData * outChannelsData;
    // std::vector<float> weightValues;
    // weightValues.reserve(weight_elem_count);
    std::vector<float> weightValues = {-0.2430,  0.1745, -0.0398,  0.1308, -0.1199, -0.1530, -0.0807,  0.2125, -0.1672, -0.0767, -0.0102, -0.0107, -0.2086,  0.0559, -0.0215,  0.0995, -0.2034, -0.0632,  0.1595, -0.0834, -0.1525,  0.0219,
            -0.0367, -0.2446, 0.1356,  0.0719,  0.0870,  0.2051, -0.1150, -0.0719,  0.1175, -0.2011 };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < weight_elem_count; ++i) {
        weightValues.push_back(dis(rd));
    }
    auto weightTensor = Ort::Value::CreateTensor<float>(mem_info, weightValues.data(), weight_elem_count, weightShape.data(), weightShape.size());

    const float* weightTensorData = weightTensor.GetTensorData<float>();

    for (size_t i = 0; i < weight_elem_count; ++i) {
        std::cout << "WeightTensor Data[" << i << "]: " << weightTensorData[i] << std::endl;
    }

    // Ort::Value weightTensor = Ort::Value::CreateTensor<int64_t>(mem_info, nullptr, weight_elem_count, weightShape.data(), weightShape.size());

    // out_spatial_shape = ops.get_conv_output_size(
    //                 spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation)
    std::vector<int64_t> out_spatial_size = get_conv_output_size(spatialShapeData, kernelSizeData, strideData, paddingData, dilationData);
    std::cout << "out_spatial_shape:\n\n";
    for (const auto& size : out_spatial_size) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    // auto datas = nullptr;

    std::cout << "Calling function call:\n";

    // std::tie(outputSliceTensor, indice_pair_outputTensor, outputTensor) = getIndicesPair(context);

    // std::vector<Ort::Value> ort_inputs;
    auto gip_outs = indicesPairOperator.getIndicesPair(context);

    std::cout << "\nReturned!\n";

    int64_t numActOut = std::get<0>(gip_outs);
    std::vector<int64_t> slice_out_data = std::get<1>(gip_outs);
    std::vector<int64_t> slice_out_shape = std::get<2>(gip_outs);
    std::vector<int64_t> indicespair_data = std::get<3>(gip_outs);
    std::vector<int64_t> indicespair_shape = std::get<4>(gip_outs);
    std::vector<int64_t> indicesnum_data = std::get<5>(gip_outs);
    std::vector<int64_t> indicesnum_shape = std::get<6>(gip_outs);


    auto SliceOutTensor = Ort::Value::CreateTensor<int64_t>(mem_info, slice_out_data.data(), slice_out_data.size(), slice_out_shape.data(), slice_out_shape.size());

    auto indicesPairTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indicespair_data.data(), indicespair_data.size(), indicespair_shape.data(), indicespair_shape.size());

    auto indicesNumTensor = Ort::Value::CreateTensor<int64_t>(mem_info, indicesnum_data.data(), indicesnum_data.size(), indicesnum_shape.data(), indicesnum_shape.size());


    // std::cout << "indicesPairData Updated from GIP OP: ";
    int64_t* indice_pair_data = indicesPairTensor.GetTensorMutableData<int64_t>();
    const auto indicesPairShape = indicesPairTensor.GetTensorTypeAndShapeInfo().GetShape();
    // const size_t indice_pair_fill_size = indicesPairTensor.GetTensorTypeAndShapeInfo().GetElementCount();
    // std::cout << "indice_pair_fill_size from GIP OP: " << indice_pair_fill_size << "\n";
    // for (size_t i = 0; i < indice_pair_fill_size; ++i) {
    //     std::cout << indice_pair_updated[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << "Indicespair tensor shape: ";
    for (size_t i = 0; i < indicesPairShape.size(); ++i) {
        std::cout << indicesPairShape[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "IndicesNum tensor shape: ";
    // std::vector<Ort::Value>& ort_inputs
    const int64_t* indice_num_updated = indicesNumTensor.GetTensorData<int64_t>();
    const auto IndiceNumShape = indicesNumTensor.GetTensorTypeAndShapeInfo().GetShape();
    for (size_t i = 0; i < IndiceNumShape.size(); ++i) {
        std::cout << IndiceNumShape[i] << " ";
    }
    std::cout << std::endl;

    // Print values
    std::cout << "IndicesPair tensor values: ";
    const size_t indice_pair_size = indicesPairTensor.GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < indice_pair_size; ++i) {
        std::cout << indice_pair_data[i] << " ";
    }
    std::cout << std::endl;

    // Create input buffer and Output buffer
    auto ndim = weightShape.size() - 2;
    auto kernelVolume = indicesPairShape[0];
    auto numInPlanes = featuresShape[1];
    auto numOutPlanes = weightShape[ndim + 1];

    //Move indicesNum => CPU
    // Define iterator
    auto indicePairMaxSizeIter = std::max_element(indice_num_updated, indice_num_updated + kernelVolume);
    auto indicePairMaxOffset = indicePairMaxSizeIter - indice_num_updated;
    int64_t indicePairMaxSize = *indicePairMaxSizeIter;

    std::cout << "numInPlanes:" << numInPlanes << "\n";
    std::cout << "numOutplanes:" << numOutPlanes << "\n";
    std::cout << "Maximum value of indice_num_updated: " << indicePairMaxSize << std::endl;
    std::cout << "indicePairMaxOffset:" << indicePairMaxOffset << "\n";

    // torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
    // torch::Tensor inputBuffer =
    //     torch::zeros({indicePairMaxSize, numInPlanes}, options);
    // torch::Tensor outputBuffer =
    //     torch::zeros({indicePairMaxSize, numOutPlanes}, options);
    // filters = filters.view({-1, numInPlanes, numOutPlanes});

    int64_t output_fill_value = 0;
    std::vector<int64_t> OutputVector(numActOut * numOutPlanes);
    std::fill(OutputVector.begin(), OutputVector.end(), output_fill_value);

    std::vector<int64_t> outputShape = {numActOut, numOutPlanes};
    // size_t output_element_count = numActOut* numOutPlanes;
    auto outputholder = Ort::Value::CreateTensor<int64_t>(mem_info, OutputVector.data(), OutputVector.size(), outputShape.data(), outputShape.size());

    int64_t inputBuffer_fill_value = 0;
    std::vector<int64_t> inputBufferVector(indicePairMaxSize * numInPlanes);
    std::fill(inputBufferVector.begin(), inputBufferVector.end(), inputBuffer_fill_value);

    std::vector<int64_t> inputBufferShape = {indicePairMaxSize, numInPlanes};
    // size_t inputBuffer_element_count = indicePairMaxSize* numInPlanes;
    auto inputBuffer = Ort::Value::CreateTensor<int64_t>(mem_info, inputBufferVector.data(), inputBufferVector.size(), inputBufferShape.data(), inputBufferShape.size());

    int64_t outputBuffer_fill_value = 0;
    std::vector<int64_t> outputBufferVector(indicePairMaxSize* numOutPlanes);
    std::fill(outputBufferVector.begin(), outputBufferVector.end(), outputBuffer_fill_value);

    std::vector<int64_t> outputBufferShape = {indicePairMaxSize, numOutPlanes};
    // size_t outputBuffer_element_count = indicePairMaxSize* numOutPlanes;
    auto outputBuffer = Ort::Value::CreateTensor<int64_t>(mem_info, outputBufferVector.data(), outputBufferVector.size(), outputBufferShape.data(), outputBufferShape.size());

    // // filters = filters.view({-1, numInPlanes, numOutPlanes});
    std::cout << "CREATE FILTER TENSOR:\n";
    float* filterDataAccess = weightTensor.GetTensorMutableData<float>();
    std::vector<int64_t> filtersShape = {kernelVolume, numInPlanes, numOutPlanes};
    size_t filters_elem_count = kernelVolume * numInPlanes * numOutPlanes;
    // std::vector<int64_t> filterData(filters_elem_count);
    auto filterTensor = Ort::Value::CreateTensor<float>(mem_info, filterDataAccess, filters_elem_count, filtersShape.data(), filtersShape.size());
    // Ort::Value filtersTensor = Ort::Value::CreateTensor<int64_t>(mem_info, filterData, filters_elem_count, filtersShape.data(), filtersShape.size());

    const float* filterData = filterTensor.GetTensorData<float>();
    for (size_t j = 0; j < filters_elem_count; ++j) {
        std::cout << "filterTensor Creation from weight[" << j << "]: " << filterData[j] << "\n";
    }
    std::cout << "\n";

    for (int64_t i = 0; i < kernelVolume; ++i) {
      auto nHot = indice_num_updated[i];
      std::cout << "nHot:" << nHot << "\n";
      if (nHot <= 0 || (subm_bool && i == indicePairMaxOffset)) {
        std::cout << "nHot and indicesPairMaxOffset:" << nHot << "    " << indicePairMaxOffset << "\n";
        continue;
      }

    std::cout << "Preparing tensorView tensors\n";
    auto inputBufferTensorView = tv::ort2tv<int64_t>(inputBuffer);
    auto featuresTensorView = tv::ort2tv<const int64_t>(featuresData, featuresShape);
    // auto featuresTensorView = tv::ort2tv<int64_t>(features_tensor);
    auto indicesPairTensorView = tv::ort2tv<const int64_t>(indice_pair_data, indicespair_shape).subview(static_cast<int>(i), static_cast<int>(inverse_bool));
    // auto indicesPairTensorView = tv::ort2tv<int64_t>(indicesPairTensor).subview(static_cast<int>(i), static_cast<int>(inverse_bool));

    //Prepare TensorViewInputs
    std::cout << "inputBufferTensorView:" << std::endl;
    tv::printTensorView(inputBufferTensorView);
    std::cout << std::endl;

    // Print featuresTensorView
    std::cout << "featuresTensorView:" << std::endl;
    tv::printTensorView(featuresTensorView);
    std::cout << std::endl;

    // Print indicesPairTensorView
    std::cout << "indicesPairTensorView:" << std::endl;
    tv::printTensorView(indicesPairTensorView);
    std::cout << std::endl;

    auto gatherFtor = functor::SparseGatherFunctor<tv::CPU, int64_t, int64_t>();

    gatherFtor(inputBufferTensorView,
        featuresTensorView, indicesPairTensorView, nHot);

    std::cout << "Updated inputBufferTensorView After Functor:" << std::endl;
    tv::printTensorView(inputBufferTensorView);
    std::cout << std::endl;

    std::cout << "Updated indicesPairTensorView After Functor:" << std::endl;
    tv::printTensorView(indicesPairTensorView);
    std::cout << std::endl;

    // torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
    //Calling GEMM kernelop

    // std::cout << "filter_i size:" << filterData + (i * numInPlanes * numOutPlanes) << "\n\n";

    std::vector<int64_t> filterTensor_iShape = {numInPlanes, numOutPlanes};
    size_t filterTensor_i_elem_count = numInPlanes * numOutPlanes;
    std::vector<float> filterIndexData(filterTensor_i_elem_count);
    // auto filter_iData = filterData + (i * filterTensor_i_elem_count);
    std::copy(filterTensor.GetTensorMutableData<float>() + (i * numInPlanes * numOutPlanes), filterTensor.GetTensorMutableData<float>() + ((i + 1) * numInPlanes * numOutPlanes), filterIndexData.begin());
    auto indexedFilterTensor = Ort::Value::CreateTensor<float>(mem_info, filterIndexData.data(), filters_elem_count / kernelVolume, filterTensor_iShape.data(), filterTensor_iShape.size());
    // Ort::Value filter_iTensor = Ort::Value::CreateTensor<int64_t>(mem_info, filter_iData, filterTensor_i_elem_count, filterTensor_iShape.data(), filterTensor_iShape.size());
    // auto filterTensor_i = Ort::Value::CreateTensor<int64_t>(
    //     mem_info, filterData + (i * numInPlanes * numOutPlanes), filterTensor_i_elem_count,
    //     filterTensor_iShape.data(), filterTensor_iShape.size());

    const float* filter_iDataAccess = indexedFilterTensor.GetTensorData<float>();
    for (size_t j = 0; j < filterTensor_i_elem_count; ++j) {
        std::cout << "filterTensor[" << i << "][" << j << "]: " << filter_iDataAccess[j] << "\n";
    }
    std::cout << "\n";

    // auto filterTensor_iFloat = Ort::Value::CreateTensor<float>(mem_info, reinterpret_cast<float*>(filterData + (i * numInPlanes * numOutPlanes)), filterTensor_i_elem_count, filterTensor_iShape.data(), filterTensor_iShape.size());
    // const float* floatData = filterTensor_iFloat.GetTensorData<float>();
    // std::cout << "filters[i]:";
    // for (size_t j = 0; j < filterTensor_i_elem_count; ++j) {
    //     std::cout << floatData[j] << " ";
    // }
    // std::cout << "\n\n";

    // auto outputBufferBlob =
    //         torch::from_blob(outputBuffer.data<T>(), {nHot, numOutPlanes}, options);
    //     auto inputBufferBlob =
    //         torch::from_blob(inputBuffer.data<T>(), {nHot, numInPlanes}, options);

    std::vector<int64_t> inputBufferBlobShape = {nHot, numInPlanes};
    size_t inputBufferBlob_element_count = nHot* numInPlanes;
    std::vector<float> inputBufferData(inputBufferBlob_element_count);
    std::copy(inputBuffer.GetTensorMutableData<int64_t>(), inputBuffer.GetTensorMutableData<int64_t>() + inputBufferBlob_element_count, inputBufferData.begin());
    auto inputBufferBlob = Ort::Value::CreateTensor<float>(mem_info, inputBufferData.data(), inputBufferBlob_element_count, inputBufferBlobShape.data(), inputBufferBlobShape.size());

    const float* floatData = inputBufferBlob.GetTensorData<float>();

    std::cout << "inputBufferBlob data: ";
    for (size_t i = 0; i < inputBufferBlob_element_count; ++i) {
        std::cout << floatData[i] << " ";
    }
    std::cout << "\n";


    std::vector<int64_t> outputBufferBlobShape = {nHot, numOutPlanes};
    size_t outputBufferBlob_element_count = nHot* numOutPlanes;

    std::vector<float> outputBufferBlobVectorData(outputBufferBlob_element_count);
    // Copy the data from the int64 outputBuffer to the float outputBufferBlobData
    std::copy(outputBuffer.GetTensorMutableData<int64_t>(), outputBuffer.GetTensorMutableData<int64_t>() + outputBufferBlob_element_count, outputBufferBlobVectorData.begin());


    auto outputBufferBlob = Ort::Value::CreateTensor<float>(mem_info, outputBufferBlobVectorData.data(), outputBufferBlob_element_count, outputBufferBlobShape.data(), outputBufferBlobShape.size());

    const float* outputBufferBlobData = outputBufferBlob.GetTensorData<float>();

    std::cout << "outputBufferBlob data: ";
    for (size_t i = 0; i < outputBufferBlob_element_count; ++i) {
        std::cout << outputBufferBlobData[i] << " ";
    }
    std::cout << "\n";

    const Ort::Value gemm_inputs[2] = {std::move(inputBufferBlob), std::move(indexedFilterTensor)};


    Ort::Value gemm_outputs[1] = {std::move(outputBufferBlob)};
    op_gemm_.Invoke(context, gemm_inputs, 2, gemm_outputs, 1);

    std::cout << "GEMM outputData: ";
    auto gemm_result = gemm_outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < outputBufferBlob_element_count; ++i) {
        std::cout << gemm_result[i] << " ";
    }
    std::cout << "\n";

    float* bufferblobdata = outputBuffer.GetTensorMutableData<float>();
    for (size_t i = 0; i < outputBufferVector.size(); ++i) {
        std::cout << bufferblobdata[i] << " ";
    }
    std::cout << "\n";

    std::cout << "GEMM Output Shape: [";
    for (size_t i = 0; i < outputBufferBlobShape.size(); ++i) {
      std::cout << outputBufferBlobShape[i] << "    ";
    }
    std::cout << "]\n";

    // functor::SparseScatterAddFunctor<tv::CPU, T, int> scatterFtor;
    auto scatterAddFtor = functor::SparseScatterAddFunctor<tv::CPU, float, int64_t>();

    //Prepare the ScatterAdd TensorView
    auto outputTensorView = tv::ort2tv<float>(outputholder);
    auto outputBufferTensorView = tv::ort2tv<const float>(gemm_result, outputBufferBlobShape);
    // auto outputBufferTensorView = tv::ort2tv<const int64_t>(outputBufferBlob);
    // auto indicesPairSAddTensorView = indicesPairTensorView.subview(static_cast<int>(i), static_cast<int>(!inverse_bool));
    auto indicesPairSAddTensorView = tv::ort2tv<const int64_t>(indice_pair_data, indicespair_shape).subview(static_cast<int>(i), static_cast<int>(!inverse_bool));
    // std::cout << "indicesPairSAddTensorView:" << std::endl;
    // tv::printTensorView(indicesPairSAddTensorView);
    // std::cout << std::endl;


    //Call ScatterAdd
    scatterAddFtor(outputTensorView, outputBufferTensorView, indicesPairSAddTensorView, nHot);

    std::cout << "Updated outputTensorview Final:" << std::endl;
    tv::printTensorView(outputTensorView);
    std::cout << std::endl;

    }

    SparseConvTensor out_tensor(outputholder, SliceOutTensor, out_spatial_size, batchSizeData, grid_tensor);

    // setup Output
    auto output1 = ctx.GetOutput(0, outputShape);
    const auto* OutputTensorData = outputholder.GetTensorData<float>();
    float* out1 = output1.GetTensorMutableData<float>();

    const size_t size = output1.GetTensorTypeAndShapeInfo().GetElementCount();

    for (size_t i = 0; i < size; i++) {
      out1[i] = OutputTensorData[i];
    }
    }

    // setup Output2
    // auto output2 = ctx.GetOutput(1, slice_out_shape);
    // const auto* OutputTensorData2 = SliceOutTensor.GetTensorData<float>();
    // float* out2 = output2.GetTensorMutableData<float>();

    // const size_t size2 = output2.GetTensorTypeAndShapeInfo().GetElementCount();

    // for (size_t i = 0; i < size2; i++) {
    //   out2[i] = OutputTensorData[i];
    // }
    // }


  private:
    Ort::KernelInfo info_copy_{nullptr};
    Ort::Op op_gemm_{nullptr};
    Ort::Op op_cast_{nullptr};
    IndicesPairKernel get_indices_pair_op_;
    GetIndicesPairOperator indicesPairOperator;
};

struct SparseConvolution : Ort::CustomOpBase<SparseConvolution, SparseConvKernel> {
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new SparseConvKernel(info); };

  const char* GetName() const { return "SparseConvolution"; };

  // size_t GetInputTypeCount() const { return 11; };
  size_t GetInputTypeCount() const {return 14; };

  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };

  // size_t GetOutputTypeCount() const { return 3; };
  size_t GetOutputTypeCount() const { return 1; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };
};

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const GetIndicesPair c_GetIndicesPair;
  static const SparseConvolution c_SparseConvolution;
  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_GetIndicesPair);
    domain.Add(&c_SparseConvolution);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}


IndicesPairKernel::~IndicesPairKernel() {
}

// SparseConvKernel::~SparseConvKernel() {
// }
