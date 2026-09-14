#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

// vulkantuner.h -- the CONTROL PLANE of the Vulkan runtime tuner (T3).
//
// WHY THIS FILE EXISTS (t3-shell-plan.md section 9)
// The OpenCL tuner is clean because its measurement can move out of the backend.  Ours
// cannot: a Vulkan pipeline is a runtime object (specialisation constants, an `align`
// side effect on kpad/aligned, a pre-record replay bypass), so the tuner has to live
// inside the process.  What we buy instead is a NARROW interface: the backend exposes
// four functions and nothing else, and every piece of tuner policy -- which candidates
// exist, which are legal, which gate applies, how they are ranked, where the cache lives
// -- lives here (t3-shell-plan.md section 9.1 R1/R5).
//
// The dividing line, mechanically enforced by t3-guardrails.py:
//   * this side never touches a `vk*` type, a pipeline handle or a device object;
//   * the backend never learns what a "candidate" is.
//
// Logging goes to std::cerr, like the backend's T2/T3 probes.  That is deliberate: the
// acceptance for this cut is a readable ORDER of "decision -> install -> frozen ->
// pr_recording", and three lines in three different streams is not readable order.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "ggml-backend.h"
#include "vulkan-tile-validity.h"

// The backend interface (defined in the vendored ggml-vulkan.cpp).  Declared HERE, at
// global scope with the C++ linkage the definition has, and NOT in ggml-vulkan.h: that
// header is vendored public API and the ggml CMake surface stays untouched
// (t3-shell-plan.md section 7.1-7).  Four functions total (R1).
//
// Returns the written length (excluding the terminator) or 0 when the backend is not a
// Vulkan one / the buffer is too small.
size_t ggml_backend_vk_tune_fingerprint(ggml_backend_t backend, char* out, size_t outSize);
// Installs a row set into ONE context.  Returns false and fills `err` on refusal; the
// backend logs the refusal loudly itself.
bool ggml_backend_vk_tune_install_rows(ggml_backend_t backend, const vk_tune_row_spec* rows, size_t nRows, char* err, size_t errSize);

namespace VulkanTuner {

  // One tune decision, taken at most once per (evaluator, device) and reused by every
  // context on that device (t3-shell-plan.md section 2).  Immutable once built.
  struct Decision {
    int deviceIdx = 0;
    std::string deviceName;
    std::string fingerprint;      // full device/driver fingerprint (cache header identity)
    std::string deviceFile;       // sha256(fingerprint)[0:16].txt  (filename-safe, section 1)
    std::string cacheDir;         // <homeDataDir>/vulkan-tune
    std::string cachePath;        // cacheDir + "/" + deviceFile (may not exist yet)
    bool cacheFileExists = false;
    bool imported = false;        // row set came from KATAGO_VK_TUNE_IMPORT
    std::string importedFrom;     // path the import came from
    // Cache-growth accounting (t1-schema.md section 1, "pipeline cache growth"): the
    // bytes this process loaded from the persistent pipeline cache.  The "after" reading
    // only exists at the NEXT startup, so a single run can report the "before" half.
    bool hasPipelineCacheBytes = false;
    size_t pipelineCacheBytesBefore = 0;
    size_t rows = 0;              // rows handed to the backend (1b-2b: always 0)
    double tuneMs = 0.0;          // wall time of this decision, never mixed into visits/s
    std::string note;             // one short human-readable clause for the log line
  };

  // THE decision entry point.  Returns nullptr when the tuner is off -- including the
  // KATAGO_VK_TUNE=0 case, which must stay a complete no-op except for one loud line if
  // KATAGO_VK_TUNE_IMPORT was also requested (t3-shell-plan.md section 6.1 item 7).
  // Never throws: a failure here must fail open (behave exactly like "no tuned rows") and
  // say so loudly, never take the process down (section 3).
  std::shared_ptr<Decision> makeDecision(
    ggml_backend_t backend,
    int deviceIdx,
    const std::string& homeDataDirOverride,
    int nnXLen,
    int nnYLen,
    int maxBatchSize
  );

  // Per-context install.  Called once per ComputeHandle, after the backend exists and
  // before the first graph build; idempotent by construction (the context is fresh).
  void installForHandle(const std::shared_ptr<Decision>& decision, ggml_backend_t backend);

}  // namespace VulkanTuner

#endif  // NEURALNET_VULKANTUNER_H_
