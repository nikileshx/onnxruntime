#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>

#include "core/common/common.h"
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);
#endif

static const char* c_OpDomain = "test.customop";

struct KernelOne {

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    auto input_Y = ctx.GetInput(1);
    const float* X = input_X.GetTensorData<float>();
    const float* Y = input_Y.GetTensorData<float>();

    // Setup output
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

    auto output = ctx.GetOutput(0, dimensions);
    float* out = output.GetTensorMutableData<float>();

    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

    // Do computation
#ifdef USE_CUDA
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());
    cuda_add(size, out, X, Y, stream);
#else
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
#endif
  }
};

struct KernelTwo {

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    const float* X = input_X.GetTensorData<float>();

    // Setup output
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

    auto output = ctx.GetOutput(0, dimensions);
    int32_t* out = output.GetTensorMutableData<int32_t>();

    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

    // Do computation
    for (size_t i = 0; i < size; i++) {
      out[i] = static_cast<int32_t>(round(X[i]));
    }
  }
};

struct KernelThree {

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    auto input_Y = ctx.GetInput(1);
    const float* X = input_X.GetTensorData<float>();
    const float* Y = input_Y.GetTensorData<float>();
    std::cout << "INSIDE KERNEL THREE NATIVE OP\n";

    // Setup output
    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

    auto output = ctx.GetOutput(0, dimensions);
    auto output2 = ctx.GetOutput(1, dimensions);
    float* out = output.GetTensorMutableData<float>();
    float* out2 = output2.GetTensorMutableData<float>();

    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

#ifdef USE_CUDA
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());
    cuda_add(size, out, X, Y, stream);
#else
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
      out2[i] = X[i];
    }
#endif
    }
};

struct CustomNativeKernel {
  CustomNativeKernel(const OrtKernelInfo* info) {
    Ort::ConstKernelInfo k_info{info};
    info_copy_ = k_info.Copy();

    // Create the addition operation
    // const char* add_type_constraint_names[1] = {"T"};
    // ONNXTensorElementDataType add_type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

    const char* gather_type_constraint_names[2] = {"T", "Tind"};
    ONNXTensorElementDataType gather_type_constraint_values[2] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};

    constexpr int64_t axis_value = 0;
    auto axis = Ort::OpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

    Ort::OpAttr attr_list[1] = {std::move(axis)};

    op_gather_ = Ort::Op::Create(info_copy_, "Gather", "", 13,
                          gather_type_constraint_names,
                          gather_type_constraint_values,
                          2, attr_list, 1, 2, 1);

    // // Save the OrtApi pointer
    // api_ = api;
  }

  ~CustomNativeKernel() {}

  void Compute(OrtKernelContext* context) {
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::KernelContext ctx(context);
    auto input_X = ctx.GetInput(0);
    auto input_Y = ctx.GetInput(1);

    auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();

    auto output = ctx.GetOutput(0, dimensions);
    auto output2 = ctx.GetOutput(1, dimensions);

    sub_op_.Compute(context);


    int64_t* outputData = output.GetTensorMutableData<int64_t>();
    int64_t* outputData2 = output2.GetTensorMutableData<int64_t>();

    std::cout << "OutputData: ";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << "\n";

    std::cout << "OutputData2: ";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        std::cout << outputData2[i] << " ";
    }
    std::cout << "\n";

    // ========================================================

    const OrtValue* add_inputs[2] = {output, input_Y};
    OrtValue* add_outputs[1] = {output};

    op_gather_.Invoke(context, add_inputs, 2, add_outputs, 1);
  }

 private:
  Ort::KernelInfo info_copy_{nullptr};
  Ort::Op op_gather_{nullptr};
  KernelThree sub_op_;
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelOne>().release();
  };

  const char* GetName() const { return "CustomOpOne"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

};

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<CustomOpTwo>().release();
  };

  const char* GetName() const { return "CustomOpTwo"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

};

struct CustomOpThree : Ort::CustomOpBase<CustomOpThree, KernelThree> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelThree>().release();
  };

  const char* GetName() const { return "CustomOpThree"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

struct CustomNativeOp : Ort::CustomOpBase<CustomNativeOp, CustomNativeKernel> {
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new CustomNativeKernel(info); };
  const char* GetName() const { return "CustomNativeOp"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const CustomOpOne c_CustomOpOne;
  static const CustomOpTwo c_CustomOpTwo;
  static const CustomOpThree c_CustomOpThree;
  static const CustomNativeOp c_CustomNativeOp;
  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CustomOpOne);
    domain.Add(&c_CustomOpTwo);
    domain.Add(&c_CustomOpThree);
    domain.Add(&c_CustomNativeOp);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } ORT_CATCH (const std::exception& e) {
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


// CustomNativeKernel::~CustomNativeKernel() {
// }
