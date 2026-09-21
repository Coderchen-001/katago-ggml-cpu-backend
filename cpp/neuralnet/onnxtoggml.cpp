// Out-of-line definitions for the shared ONNX -> ggml layer. See onnxtoggml.h.

#include "onnxtoggml.h"

#if defined(USE_VULKAN_BACKEND) || defined(USE_CPUGGML_BACKEND)

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

#endif // USE_VULKAN_BACKEND || defined(USE_CPUGGML_BACKEND)
