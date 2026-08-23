// Vulkan backend for KataGo (Phase 1A skeleton).
//
// Runs neural net inference on any Vulkan-capable GPU, using the ggml Vulkan
// backend (ggml-vulkan) as the kernel base. The model format is a .onnx file
// dumped by 'katago dumponnx' (or a .bin.gz, which the backend converts to the
// same ONNX graph in memory). This file currently implements the backend
// plumbing: device enumeration and the nninterface.h entry points. Model
// loading and inference arrive in Phase 1B.
//
// IO tensor protocol (identical to the ONNX/TensorRT path): four NCHW float32
// inputs InputSpatial / InputGlobal / InputMeta / InputMask and five NCHW
// float32 outputs OutputPolicyPass / OutputPolicy / OutputValue /
// OutputScoreValue / OutputOwnership, all raw logits. See onnxbackend.cpp for
// the post-processing contract that getOutput must reproduce.

#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"

#include <vulkan/vulkan.h>

#include <iostream>
#include <set>
#include <vector>

using namespace std;

//--------------------------------------------------------------

struct LoadedModel {
  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  (void)expectedSha256;
  // Phase 1A skeleton: model loading is not implemented yet.
  throw StringError(
    "VULKAN backend: model loading is not implemented in this Phase 1A skeleton build. "
    "File: " + file);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  (void)loadedModel;
  throw StringError("VULKAN backend: getModelDesc is not implemented in this Phase 1A skeleton build.");
}

//--------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;

  ComputeContext(int xLen, int yLen)
    : nnXLen(xLen), nnYLen(yLen)
  {}

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)homeDataDirOverride;
  (void)useFP16Mode;
  (void)loadedModel;
  (void)cfg;

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen);
  if(logger != NULL)
    logger->write("VULKAN backend: creating compute context for " +
                  Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen));
  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* ctx;

  ComputeHandle(ComputeContext* context)
    : ctx(context)
  {}

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)loadedModel;
  (void)logger;
  (void)maxBatchSize;
  (void)requireExactNNLen;
  (void)gpuIdxForThisThread;
  (void)serverThreadIdx;
  if(inputsUseNHWC)
    throw StringError("VULKAN backend: inputsUseNHWC = true not supported, must use NCHW");

  return new ComputeHandle(context);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}

//--------------------------------------------------------------

struct InputBuffers {
  InputBuffers() {}
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  (void)maxBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  return new InputBuffers();
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//--------------------------------------------------------------

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

//--------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  (void)gpuHandle;
  (void)inputBuffers;
  (void)numBatchEltsFilled;
  (void)inputBufs;
  (void)outputs;
  throw StringError("VULKAN backend: getOutput is not implemented in this Phase 1A skeleton build.");
}

//--------------------------------------------------------------

std::string NeuralNet::getRuntimeBackendDetail(ConfigParser& cfg) {
  (void)cfg;
  return "vulkan";
}

int NeuralNet::getNumEffectiveDevices(ConfigParser& cfg, const std::vector<int>& gpuIdxByServerThread) {
  (void)cfg;
  std::set<int> distinctDevices;
  for(int idx : gpuIdxByServerThread)
    distinctDevices.insert(idx);
  return std::max(1, (int)distinctDevices.size());
}

NeuralNet::BatchPolicy NeuralNet::getBatchPolicy(ConfigParser& cfg) {
  (void)cfg;
  // ggml-vulkan schedules at runtime with no per-shape recompilation, so any
  // batch size is fine and larger ones amortize per-call overhead.
  return NeuralNet::BatchPolicy::Dynamic;
}

//--------------------------------------------------------------

static const char* physicalDeviceTypeString(VkPhysicalDeviceType type) {
  switch(type) {
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
  case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
  default:                                     return "other";
  }
}

void NeuralNet::printDevices() {
  VkInstance instance = VK_NULL_HANDLE;
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "KataGo";
  appInfo.applicationVersion = 1;
  appInfo.pEngineName = "KataGo";
  appInfo.engineVersion = 1;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  if(result != VK_SUCCESS) {
    cout << "Vulkan backend: could not create a Vulkan instance (is a Vulkan driver installed?)" << endl;
    return;
  }

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  if(deviceCount == 0)
    cout << "Vulkan backend: no Vulkan devices found" << endl;
  for(uint32_t i = 0; i < deviceCount; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);
    cout << "Vulkan device " << i << ": " << props.deviceName
         << " (vendor " << Global::uint64ToString((uint64_t)props.vendorID)
         << ", device " << Global::uint64ToString((uint64_t)props.deviceID)
         << ", " << physicalDeviceTypeString(props.deviceType)
         << ", apiVersion " << (props.apiVersion >> 22) << "."
         << ((props.apiVersion >> 12) & 0x3FF) << "."
         << (props.apiVersion & 0xFFF) << ")" << endl;
  }

  vkDestroyInstance(instance, nullptr);
}

//--------------------------------------------------------------
// The layer-level test entry points are not implemented for this backend.

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool,
  const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

#endif  // USE_VULKAN_BACKEND
