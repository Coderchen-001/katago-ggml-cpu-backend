#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

namespace VulkanTuner {

  namespace {

    const char* const LOG_PREFIX = "KataGoVulkanTuner: [T3] ";

    // Schema version of anything this file reads or writes.  Bump it when the MEANING of
    // a field changes: a cache written by an older tuner must not be read as if it had
    // today's meaning (t1-schema.md section 1, "VERSION").
    const int TUNER_VERSION = 1;

    // KATAGO_VK_TUNE: unset or anything other than "0" = enabled.  "0" is the reference
    // configuration and must stay bit-identical to stock, which is why every entry point
    // below starts with this check and returns before doing anything observable.
    bool tuneEnabled() {
      const char* env = getenv("KATAGO_VK_TUNE");
      return !(env != nullptr && strcmp(env, "0") == 0);
    }

    string shortHash(const string& s) {
      char hash[65];
      SHA2::get256(reinterpret_cast<const uint8_t*>(s.data()), s.size(), hash);
      return string(hash).substr(0, 16);
    }

    // Size of the persistent pipeline cache this process LOADED (t1-schema.md section 1,
    // "pipeline cache growth").  Reading the env var rather than asking the backend keeps
    // the interface at four functions (R1) and keeps this purely an accounting question.
    // The "after" half of the accounting can only be read at the next startup, so it is
    // not reported here -- reporting a same-run number as "after" would be fiction.
    bool pipelineCacheBytesBefore(size_t& out) {
      const char* path = getenv("GGML_VK_PIPELINE_CACHE");
      if (path == nullptr || path[0] == '\0') {
        return false;
      }
      ifstream in;
      if (!FileUtils::tryOpen(in, path, ios_base::in | ios_base::binary)) {
        return false;
      }
      in.seekg(0, ios_base::end);
      const streamoff len = in.tellg();
      if (len < 0) {
        return false;
      }
      out = (size_t)len;
      return true;
    }

    // The IMPORT file format skeleton (t1-schema.md section 1: EXPORT writes this format
    // and IMPORT reads it back).  1b-2b only needs the header identity: the row section is
    // parsed in T4, and saying so loudly is the point -- an import that silently loaded
    // nothing would be indistinguishable from one that worked.
    struct ImportHeader {
      bool foundFingerprint = false;
      string fingerprint;
      bool foundVersion = false;
      int version = 0;
      size_t rowLines = 0;
    };

    bool readImportHeader(const string& path, ImportHeader& out, string& err) {
      ifstream in;
      if (!FileUtils::tryOpen(in, path, ios_base::in)) {
        err = "cannot open " + path;
        return false;
      }
      string line;
      while (getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
          line.pop_back();
        }
        if (line.empty()) {
          continue;
        }
        if (line.compare(0, 8, "VERSION ") == 0) {
          out.foundVersion = true;
          out.version = atoi(line.c_str() + 8);
        }
        else if (line.compare(0, 12, "FINGERPRINT ") == 0) {
          out.foundFingerprint = true;
          out.fingerprint = line.substr(12);
        }
        else if (line.compare(0, 4, "ROW ") == 0) {
          out.rowLines++;
        }
      }
      return true;
    }

    string formatDecisionLine(const Decision& d, int nnXLen, int nnYLen, int maxBatchSize) {
      ostringstream os;
      os << LOG_PREFIX << "decision"
         << " device=" << d.deviceName
         << " deviceIdx=" << d.deviceIdx
         << " S=" << nnXLen << "x" << nnYLen
         << " mb=" << maxBatchSize
         << " rows=" << d.rowCount
         << " source=" << (d.imported ? "import" : "none")
         << " tune_ms=" << d.tuneMs
         << " cache=" << d.deviceFile
         << " cache_exists=" << (d.cacheFileExists ? 1 : 0)
         << " pipeline_cache_bytes_before="
         << (d.hasPipelineCacheBytes ? to_string(d.pipelineCacheBytesBefore) : string("unknown"))
         << " fingerprint=" << d.fingerprint
         << " (" << d.note << ")";
      return os.str();
    }

  }  // namespace

  size_t backendFingerprint(ggml_backend_t backend, char* out, size_t outSize) {
    return ::ggml_backend_vk_tune_fingerprint(backend, out, outSize);
  }

  bool backendInstallRows(ggml_backend_t backend, const vk_tune_row_spec* rows, size_t nRows, string& err) {
    vector<char> errBuf(512, '\0');
    const bool ok = ::ggml_backend_vk_tune_install_rows(backend, rows, nRows, errBuf.data(), errBuf.size());
    if (!ok) {
      err = string(errBuf.data());
    }
    return ok;
  }

  shared_ptr<Decision> makeDecision(
    ggml_backend_t backend,
    int deviceIdx,
    const string& homeDataDirOverride,
    int nnXLen,
    int nnYLen,
    int maxBatchSize
  ) {
    const char* importEnv = getenv("KATAGO_VK_TUNE_IMPORT");
    const bool enabled = tuneEnabled();

    if (!enabled) {
      // Item 7: TUNE=0 is a complete no-op, EXCEPT that an import request must not be
      // dropped in silence -- "I set the env var and nothing happened" is the trap this
      // warning exists for (lesson #46).
      if (importEnv != nullptr && importEnv[0] != '\0') {
        cerr << LOG_PREFIX << "WARNING: KATAGO_VK_TUNE_IMPORT=" << importEnv
             << " was requested but KATAGO_VK_TUNE=0 disables the tuner; nothing was imported"
             << " and no tuned row will be used (reference run stays clean)" << endl;
      }
      return nullptr;
    }

    const auto t0 = chrono::steady_clock::now();
    shared_ptr<Decision> d = make_shared<Decision>();
    d->deviceIdx = deviceIdx;

    try {
      char fp[512];
      const size_t fpLen = backendFingerprint(backend, fp, sizeof(fp));
      if (fpLen == 0) {
        cerr << LOG_PREFIX << "WARNING: the backend could not describe the device; the tuner is"
             << " disabled for device " << deviceIdx << " (fail-open: no tuned rows)" << endl;
        return nullptr;
      }
      d->fingerprint = string(fp, fpLen);

      // The device name comes out of the fingerprint rather than from a second query: one
      // source, so the log line can never disagree with the identity used for the file.
      {
        const size_t at = d->fingerprint.find("|name=");
        const size_t end = d->fingerprint.find('|', at == string::npos ? 0 : at + 6);
        d->deviceName = at == string::npos
          ? string("unknown")
          : d->fingerprint.substr(at + 6, (end == string::npos ? d->fingerprint.size() : end) - (at + 6));
      }

      d->deviceFile = shortHash(d->fingerprint) + ".txt";
      d->cacheDir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkan-tune";
      MakeDir::make(d->cacheDir);
      d->cachePath = d->cacheDir + "/" + d->deviceFile;
      d->cacheFileExists = FileUtils::exists(d->cachePath);
      d->hasPipelineCacheBytes = pipelineCacheBytesBefore(d->pipelineCacheBytesBefore);

      if (importEnv != nullptr && importEnv[0] != '\0') {
        ImportHeader hdr;
        string err;
        if (!readImportHeader(importEnv, hdr, err)) {
          cerr << LOG_PREFIX << "WARNING: KATAGO_VK_TUNE_IMPORT refused: " << err
               << " (fail-open: no tuned rows)" << endl;
          return nullptr;
        }
        if (!hdr.foundFingerprint) {
          cerr << LOG_PREFIX << "WARNING: KATAGO_VK_TUNE_IMPORT refused: " << importEnv
               << " has no FINGERPRINT header (fail-open: no tuned rows)" << endl;
          return nullptr;
        }
        if (hdr.fingerprint != d->fingerprint) {
          // Section 2.1: an import is accepted only for the SAME fingerprint.  Merging
          // across devices is how a tuned row for a 5080 ends up on a 1660.
          cerr << LOG_PREFIX << "WARNING: KATAGO_VK_TUNE_IMPORT refused: fingerprint mismatch"
               << " (file=" << hdr.fingerprint << " current=" << d->fingerprint
               << "; fail-open: no tuned rows)" << endl;
          return nullptr;
        }
        if (hdr.foundVersion && hdr.version != TUNER_VERSION) {
          // The fingerprint matching is not enough: the same device read by a different
          // tuner may mean something different by the same bytes.
          cerr << LOG_PREFIX << "WARNING: KATAGO_VK_TUNE_IMPORT refused: file VERSION=" << hdr.version
               << " but this tuner speaks VERSION=" << TUNER_VERSION
               << " (fail-open: no tuned rows)" << endl;
          return nullptr;
        }
        d->imported = true;
        d->importedFrom = importEnv;
        d->note = "import fingerprint matches, row parsing arrives in T4";
      }
      else if (d->cacheFileExists) {
        d->note = "cache file present, loading arrives in T2 step 2";
      }
      else {
        d->note = "no cache file, tuning arrives in T2 step 2";
      }

      // 1b-2b: the row set is empty by construction.  Tuning/cache loading replaces this
      // in step 2 and must produce rows through the same legality table (section 1.7).
      d->rowCount = 0;
      d->tuneMs = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
      cerr << formatDecisionLine(*d, nnXLen, nnYLen, maxBatchSize) << endl;
      return d;
    }
    catch (const std::exception& e) {
      // Section 3: fail open and loud.  A tuner that can take down a search is worse than
      // no tuner at all.
      cerr << LOG_PREFIX << "WARNING: the tuner failed for device " << deviceIdx << ": " << e.what()
           << " (fail-open: no tuned rows)" << endl;
      return nullptr;
    }
  }

  void installForHandle(const shared_ptr<Decision>& decision, ggml_backend_t backend) {
    if (decision == nullptr) {
      return;
    }
    // The backend prints the per-context install line (it is the only side that can name
    // the context), so this side only has to report a failure.
    const vk_tune_row_spec* rows = nullptr;
    string err;
    if (!backendInstallRows(backend, rows, decision->rowCount, err)) {
      cerr << LOG_PREFIX << "WARNING: install failed for " << decision->deviceName << ": " << err
           << " (fail-open: this context uses the stock heuristic)" << endl;
    }
  }

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND
