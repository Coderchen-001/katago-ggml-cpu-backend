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
#include <vector>

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

  // The model half of the cache's outer key, built by the caller from the loaded model's
  // description (the shell must not learn what a LoadedModel is).  Split into the two forms
  // the format needs:
  //   * `name` goes ONLY into the file header -- a model name can contain characters that
  //     are not safe in a filename, and it is redundant with the sha for routing;
  //   * `fileTag` is the filename-safe `mv<version>-sha<sha16>` form.
  struct ModelIdentity {
    int version = 0;
    std::string sha16;   // sha256[0:16], or "none" when the model has no sha
    std::string name;    // for the header only
    // `mv<version>-sha<sha16>`: used in the file name.
    std::string fileTag() const {
      return "mv" + std::to_string(version) + "-sha" + (sha16.empty() ? std::string("none") : sha16);
    }
    // `mv<version>-sha<sha16>-name<name>`: written to the header's MODEL field.
    std::string headerValue() const {
      return fileTag() + "-name" + (name.empty() ? std::string("none") : name);
    }
  };

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
    // The rest of the cache's outer key (t1-schema.md section 4): the same device can hold
    // a different model, board size or max batch, and none of those may share a row set.
    std::string modelIdentity;    // header form (mv…-sha…-name…)
    bool negative = false;        // this key's row set is a "deterministically all-rejected" marker
    int nnXLen = 0;
    int nnYLen = 0;
    int maxBatchSize = 0;
    bool imported = false;        // row set came from KATAGO_VK_TUNE_IMPORT
    std::string importedFrom;     // path the import came from
    // The rows to hand to the backend, in the wire form both sides share.  Parsed from the
    // cache or the import file; still refused by the backend's T4 gate today (loudly), and
    // never silently dropped.
    std::vector<vk_tune_row_spec> rows;
    std::string rowSource;        // "none" / "cache" / "import"
    // "absent" / "loaded" / "negative" / "rejected"
    std::string cacheState = "absent";
    // Cache-growth accounting (t1-schema.md section 1, "pipeline cache growth"): the
    // bytes this process loaded from the persistent pipeline cache.  The "after" reading
    // only exists at the NEXT startup, so a single run can report the "before" half.
    bool hasPipelineCacheBytes = false;
    size_t pipelineCacheBytesBefore = 0;
    // How many rows were handed to the backend.  A COUNT, not the rows themselves: the
    // decision log prints it as the `rows=` field -- naming the member rowCount is what
    // keeps "rows" from reading like a row container.
    size_t rowCount = 0;
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
    int maxBatchSize,
    const ModelIdentity& model
  );

  // Per-context install.  Called once per ComputeHandle, after the backend exists and
  // before the first graph build; idempotent by construction (the context is fresh).
  void installForHandle(const std::shared_ptr<Decision>& decision, ggml_backend_t backend);

  // The ONE row-set file format (t1-schema.md section 1): the cache file, KATAGO_VK_TUNE_IMPORT
  // and KATAGO_VK_TUNE_EXPORT all use it, so a file written by one run can be read by the
  // next, by another machine, and by a hand-written fixture in a test.
  //
  //   VERSION <n>            schema version (refused on mismatch)
  //   TUNER_VERSION <n>      enumerator version (refused on mismatch)
  //   FINGERPRINT <string>   the full device/driver fingerprint (refused on mismatch)
  //   MODEL <string>         outer key part 2 (absent = not key-checked)
  //   BOARD <X>x<Y>          outer key part 3
  //   MAXBATCH <n>           outer key part 4
  //   ROWS <n>               DECLARED row count: a file that lost rows in transit is
  //                          refused instead of being read as a shorter row set
  //   ROW <path> <src0> <src1> <key_m> <key_k> <BM> <BN> <BK> <WM> <WN> <WMITER> <TM> <TN> <BS> <pad> <reshape> <smaller>
  //   NEGATIVE <reason>      this (device, model, S, mb) was DETERMINISTICALLY all-rejected;
  //                          no tuning is attempted.  Never written on a timeout, and never
  //                          written for a partial result.
  //
  // ONE FILE = ONE KEY (2026-09-13 decision): the file NAME carries the whole outer key --
  //   <sha256(fingerprint)[0:16]>-<model mvV-shaS>-<X>x<Y>-mb<N>.txt
  // -- while the header's FINGERPRINT/MODEL/BOARD/MAXBATCH stay the AUTHORITY and are still
  // checked item by item.  The name routes, the header verifies: a hand-renamed or
  // cross-machine-copied file is caught by the header, and the two cannot drift because both
  // are built from the same Decision.  Why the key is in the name rather than in per-row
  // columns: with one key per file, "re-tune" is a whole-file overwrite instead of a
  // read-modify-write (the classic way a cache gets corrupted), and only then does a
  // `benchmark -threads 1,2,4` sweep keep every bucket's rows instead of only the last
  // one's.  Precedent: OpenCLTuner::defaultFileName, which the plan says to mirror.
  //
  // Because one file = one key, ROWS and NEGATIVE are mutually exclusive STATEMENTS about
  // that key: the writer refuses to emit both (see formatRowsFileImpl).  The reader still
  // handles a hand-edited file containing both by preferring the rows and saying so loudly.
  std::string formatRowsFile(const Decision& decision);

}  // namespace VulkanTuner

#endif  // NEURALNET_VULKANTUNER_H_
