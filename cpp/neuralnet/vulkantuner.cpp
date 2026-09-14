#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"

#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// The build revision, obtained exactly the way main.cpp obtains it: a macro generated into the
// build directory.  The plan's invalidation table requires "a code change invalidates the
// row set" (t1-schema.md section 4, the buildHash row), and this is the sanction source for it;
// asking the backend instead would need a fifth interface function (R1 forbids that).
#include <program/gitinfo.h>

// For the durable flush in writeFileAtomic(): _commit (the MSVC spelling of fsync, which
// needs only <io.h> -- deliberately NOT the Win32 FlushFileBuffers, whose headers drag in
// windows.h and its macro hazards) on Windows, fsync elsewhere.
#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <stdio.h>
#else
#include <stdio.h>
#include <unistd.h>
#endif

using namespace std;

namespace VulkanTuner {

  namespace {

    const char* const LOG_PREFIX = "KataGoVulkanTuner: [T3] ";

    // ONE insertion per line, TERMINATOR INCLUDED, like the backend's ggml_vk_t3_line().
    // Two things go wrong otherwise, both measured: `<< a << b` lets another NN server
    // thread land inside this line (even inside its own prefix, which makes the line
    // unfindable), and `<< std::endl` is a SECOND insertion, so two messages can end up on
    // one physical line ("lineAlineB\n").  `LOG_PREFIX` is used HERE and nowhere else,
    // which is what makes that checkable (guardrail R10).
    void logLine(const string& msg) {
      cerr << (string(LOG_PREFIX) + msg + "\n");
    }

    // The ONE place a load refusal is worded.  Every "the file is not usable" path goes
    // through it, which (a) guarantees the wording a harness greps is stable, and (b) keeps
    // the shell-side inventory at ONE occurrence no matter how many reasons are added --
    // a new reason is a new CALL, not a new message shape.
    void refuseLoad(const string& kind, const string& why) {
      logLine("WARNING: " + kind + " refused: " + why + " (fail-open: no tuned rows)");
    }

    // Schema version of anything this file reads or writes.  Bump it when the MEANING of
    // a field changes: a cache written by an older tuner must not be read as if it had
    // today's meaning (t1-schema.md section 1, "VERSION").
    //
    // VERSION 2 (2026-09-13): the outer key moved into the FILE NAME (one file = one key) and
    // ROWS/NEGATIVE became mutually exclusive statements about that key.  There is NO
    // migration and none is needed: no production cache file existed before this (EXPORT was
    // the only writer and only tests asked for it).  A VERSION 1 file is refused loudly,
    // which is right for a file whose rows were not keyed to the name they live under.
    const int SCHEMA_VERSION = 3;
    // TUNER_VERSION = the ENUMERATOR's version: it changes when the way candidates are
    // enumerated/measured changes.  This cut adds header fields, not enumeration, so it stays
    // while SCHEMA_VERSION moves.
    const int TUNER_VERSION = 2;

    // The policy baseline (t1-schema.md section 4, "baselineVersion"): a row set measured
    // under one policy must not be reused under another.  Bump this whenever the fork's
    // DEFAULT policy changes (e.g. a new default FA tile or RMS variant) without any env
    // change -- that is exactly the case neither the build revision nor the env knobs catch
    // if a default is edited in place and the revision happens to match.
    const int TUNER_POLICY_VERSION = 1;

    // Which header keys are required depends on the CHANNEL, and the split follows what each
    // key GUARANTEES.  These two arrays are the single source of truth: the parser builds its
    // "absent" list from REQUIRED_FOR_CACHE, and R12 in t3-guardrails.py verifies both sets,
    // so moving a key between buckets is a deliberate act that has to go through the guardrail.
    //
    //   REQUIRED_ANY_CHANNEL -- the key carries a guarantee ("the policy changed, so this row
    //     set is void").  Because the parser ACCEPTS a file that is missing a merely-advisory
    //     key, putting such a key in the other bucket would make the guarantee vanish
    //     silently: the file loads, the policy changes, nothing invalidates anything.
    //   REQUIRED_FOR_CACHE -- a cache is written by us, so absence has no legitimate
    //     explanation; tolerated on IMPORT (hand-written fixtures) with a "[not key-checked]"
    //     note.  For ROWSET_HASH the tolerance goes further on IMPORT: the parsed rows are
    //     hashed live and the declared value is only a hint, since a human cannot compute it.
    const char* const REQUIRED_ANY_CHANNEL[] = { "POLICY", "BASELINE" };
    const char* const REQUIRED_FOR_CACHE[] = { "MODEL", "BOARD", "MAXBATCH", "ROWSET_HASH", "CREATED" };


    // Row-count cap.  ONE constant used by BOTH the writer and the parser (a cap that only
    // one side enforces is a cap that produces files the other side rejects): a file with
    // more rows than this is refused on read and never produced on write.  v1 is one row
    // per (family, acc), so 64 is room for the multi-row version many times over -- the
    // point of the number is to bound what a corrupt or hostile file can make us allocate.
    const size_t MAX_ROWS_PER_FILE = 64;

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

    // Make a *value* safe to place in a line-oriented text format: no newline (which would
    // forge a field), no whitespace-only confusion, no control characters.  Values come from
    // env vars and model names, i.e. from outside this program.
    string sanitizeValue(const string& in, size_t maxLen) {
      string out;
      for (char c : in) {
        if (out.size() >= maxLen) {
          break;
        }
        const unsigned char u = (unsigned char) c;
        out += (u < 0x20 || u == 0x7f) ? '_' : c;
      }
      return out;
    }

    // The policy fingerprint (t1-schema.md section 4): build revision + policy version + the
    // policy-relevant env knobs.  Deliberately assembled HERE rather than asked of the
    // backend, because the interface is capped at four functions (R1) and none of these facts
    // needs a Vulkan device to read.  The env values are the *requests*; the resolved variant
    // also depends on the device, which is already in the fingerprint/name, so the pair
    // (device, policy) is fully covered.
    string policyFingerprint() {
      static const char* const KNOBS[] = {
        "KATAGO_VK_FA", "KATAGO_VK_FA_TILE", "KATAGO_VK_FA_PRECISE", "KATAGO_VK_FA_NULL",
        "KATAGO_VK_RMS_SUBGROUP", "KATAGO_VK_M4_FUSION", "KATAGO_VK_MMPREC_F32",
        "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2", "KATAGO_VK_MM_TILE",
      };
      string s = "b" + sanitizeValue(string(GIT_REVISION), 16)
               + "|p" + std::to_string(TUNER_POLICY_VERSION);
      for (const char* knob : KNOBS) {
        const char* v = getenv(knob);
        s += "|";
        s += knob;
        s += "=";
        s += (v == nullptr) ? string("-") : sanitizeValue(string(v), 24);
      }
      return s;
    }

    // FNV-1a over the rows' identity+geometry, spelled out field by field (never over the
    // struct's bytes: padding would make the hash depend on the compiler).  This is what
    // ROWSET_HASH declares, and on the cache channel a mismatch means the file does not
    // describe the rows it contains.
    string rowSetHash(const vector<vk_tune_row_spec>& rows) {
      uint64_t h = 1469598103934665603ull;
      auto feed = [&h](uint32_t v) {
        for (int i = 0; i < 4; i++) {
          h ^= (v >> (8 * i)) & 0xffu;
          h *= 1099511628211ull;
        }
      };
      feed((uint32_t) rows.size());
      for (const vk_tune_row_spec& r : rows) {
        feed((uint32_t) r.path);
        feed((uint32_t) r.src0_type);
        feed((uint32_t) r.src1_type);
        feed(r.key_m);
        feed(r.key_k);
        feed(r.geom.BM); feed(r.geom.BN); feed(r.geom.BK);
        feed(r.geom.WM); feed(r.geom.WN); feed(r.geom.WMITER);
        feed(r.geom.TM); feed(r.geom.TN); feed(r.geom.BS);
        feed((uint32_t) r.geom.pad);
        feed((uint32_t) r.geom.reshape);
        feed((uint32_t) r.geom.smaller);
      }
      ostringstream os;
      os << hex << h;
      return os.str();
    }

    // ISO-8601 UTC, e.g. 2026-09-13T21:04:05Z.  Informational only: nothing is invalidated by
    // it (POLICY/BASELINE do that), but a human reading the cache wants to know its age.
    string createdAtNow() {
      const std::time_t now = std::time(nullptr);
      std::tm tmUtc {};
#if defined(_WIN32)
      gmtime_s(&tmUtc, &now);
#else
      gmtime_r(&now, &tmUtc);
#endif
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmUtc);
      return string(buf);
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

    // ---- the ONE row-set file format (see vulkantuner.h for the field list) ----------
    // The cache file, KATAGO_VK_TUNE_IMPORT and KATAGO_VK_TUNE_EXPORT all go through these
    // functions, so "the file the tuner writes" and "the file a human can hand-write" are
    // the same thing -- there is no second parser to drift.
    struct RowsFile {
      bool foundVersion = false;
      int version = 0;
      bool foundTunerVersion = false;
      int tunerVersion = 0;
      bool foundFingerprint = false;
      string fingerprint;
      bool foundPolicy = false;
      string policy;
      bool foundBaseline = false;
      string baseline;
      bool foundModel = false;
      string model;
      bool foundBoard = false;
      int boardX = 0, boardY = 0;
      bool foundMaxBatch = false;
      int maxBatch = 0;
      bool foundRowSetHash = false;
      string declaredRowSetHash;
      bool foundCreated = false;
      string created;
      bool foundDeclaredRows = false;
      long declaredRows = -1;
      vector<vk_tune_row_spec> rows;
      bool negative = false;
      string negativeReason;
      // Human-readable keys that were ABSENT; the caller logs them as "not key-checked"
      // rather than pretending the check happened.
      vector<string> absentKeys;
    };

    bool readWholeFile(const string& path, string& out, string& err) {
      ifstream in;
      if (!FileUtils::tryOpen(in, path, ios_base::in | ios_base::binary)) {
        err = "cannot open " + path;
        return false;
      }
      ostringstream os;
      os << in.rdbuf();
      if (in.bad()) {
        err = "read error on " + path;
        return false;
      }
      out = os.str();
      return true;
    }

    // A durable, atomic write: same-directory tmp file, flushed to the DEVICE (not just to
    // the OS cache), then renamed over the target with a few retries (Windows can fail a
    // rename while another process holds the target open).  There is no fsync helper in
    // core/FileUtils, so the flush is platform-specific -- the precedent is contribute.cpp,
    // which uses FlushFileBuffers for the same reason.
    //
    // What this guarantees: a reader never sees a half-written file, and the bytes are on
    // disk before the rename.  What it does NOT guarantee: that the rename itself is
    // durable (no directory fsync); a power loss can still lose the whole file.  That is
    // acceptable here because the file is a CACHE -- losing it costs one re-tune, which is
    // the same fail-open path as a missing file.
    bool writeFileAtomic(const string& path, const string& content, string& err) {
      // The tmp name carries the pid and a per-process sequence.  Without them two
      // processes writing the same target would share one tmp name, and A's rename could
      // publish B's bytes as A's result (the in-file outer key would then refuse it, but a
      // refusal caused by a race is a bug that should not exist).  The shell deliberately
      // does NOT use the backend's ctx ordinal here: that identity is backend-owned (R7)
      // and the shell must not invent a second numbering.
      static std::atomic<uint32_t> tmp_seq {0};
#if defined(_WIN32)
      const long pid = (long) _getpid();
#else
      const long pid = (long) getpid();
#endif
      const string tmp = path + ".tmp." + std::to_string(pid) + "."
                       + std::to_string(tmp_seq.fetch_add(1u));
      FILE* f = nullptr;
#if defined(_WIN32)
      fopen_s(&f, tmp.c_str(), "wb");
#else
      f = fopen(tmp.c_str(), "wb");
#endif
      if (f == nullptr) {
        err = "cannot open " + tmp + " for writing";
        return false;
      }
      const size_t written = content.empty() ? 0 : fwrite(content.data(), 1, content.size(), f);
      const bool wroteAll = (written == content.size());
      const bool flushed = (fflush(f) == 0);
      bool synced = true;
#if defined(_WIN32)
      synced = _commit(_fileno(f)) == 0;
#else
      synced = fsync(fileno(f)) == 0;
#endif
      fclose(f);
      if (!wroteAll) {
        err = "short write to " + tmp;
        FileUtils::tryRemoveFile(tmp);
        return false;
      }
      if (!flushed || !synced) {
        err = "could not flush " + tmp + " to disk";
        FileUtils::tryRemoveFile(tmp);
        return false;
      }
      for (int attempt = 0; attempt < 3; attempt++) {
        if (FileUtils::tryRename(tmp, path)) {
          return true;
        }
      }
      err = "could not rename " + tmp + " to " + path;
      FileUtils::tryRemoveFile(tmp);
      return false;
    }

    // Parse one ROW line.  Field count and integer syntax are checked here; the GEOMETRY is
    // checked by vk_tile_candidate_reason(), the same table the enumerator uses, so a bad
    // row is refused with the identical reason string in both places (t3-shell-plan.md
    // section 1.7 and section 6.1 item 11).
    bool parseRowLine(const string& rest, size_t lineNo, vk_tune_row_spec& out, string& err) {
      istringstream is(rest);
      int src0 = 0, src1 = 0, path = 0;
      long long keyM = 0, keyK = 0;
      long long g[12] = {0};
      string pathToken;
      if (!(is >> pathToken >> src0 >> src1 >> keyM >> keyK)) {
        err = "ROW line " + to_string(lineNo) + ": expected <path> <src0> <src1> <key_m> <key_k>";
        return false;
      }
      for (int i = 0; i < 12; i++) {
        if (!(is >> g[i])) {
          err = "ROW line " + to_string(lineNo) + ": expected 12 geometry fields after key_k";
          return false;
        }
      }
      // The path token is the vk_tile_path enum; accepting the spelling as well keeps a
      // hand-written fixture readable and costs three comparisons.
      if (pathToken == "scalar") path = VK_TILE_PATH_SCALAR;
      else if (pathToken == "cm1") path = VK_TILE_PATH_CM1;
      else if (pathToken == "cm2") path = VK_TILE_PATH_CM2;
      else path = atoi(pathToken.c_str());
      if (path != VK_TILE_PATH_SCALAR && path != VK_TILE_PATH_CM1 && path != VK_TILE_PATH_CM2) {
        err = "ROW line " + to_string(lineNo) + ": unknown path '" + pathToken + "'";
        return false;
      }
      if (keyM < 0 || keyK < 0) {
        err = "ROW line " + to_string(lineNo) + ": key_m/key_k must be >= 0 (0 = whole slot)";
        return false;
      }
      out.src0_type = src0;
      out.src1_type = src1;
      out.key_m = (uint32_t) keyM;
      out.key_k = (uint32_t) keyK;
      out.path = path;
      out.geom.BM = (uint32_t) g[0];
      out.geom.BN = (uint32_t) g[1];
      out.geom.BK = (uint32_t) g[2];
      out.geom.WM = (uint32_t) g[3];
      out.geom.WN = (uint32_t) g[4];
      out.geom.WMITER = (uint32_t) g[5];
      out.geom.TM = (uint32_t) g[6];
      out.geom.TN = (uint32_t) g[7];
      out.geom.BS = (uint32_t) g[8];
      out.geom.pad = (int) g[9];
      out.geom.reshape = (int) g[10];
      out.geom.smaller = (int) g[11];
      const char* reason = vk_tile_candidate_reason(path, out.geom);
      if (reason != nullptr) {
        err = "ROW line " + to_string(lineNo) + ": illegal geometry: " + reason;
        return false;
      }
      return true;
    }

    bool parseRowsFile(const string& text, RowsFile& out, string& err) {
      istringstream is(text);
      string line;
      size_t lineNo = 0;
      while (getline(is, line)) {
        lineNo++;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
          line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
          continue;
        }
        if (line.compare(0, 8, "VERSION ") == 0) {
          out.foundVersion = true;
          out.version = atoi(line.c_str() + 8);
        }
        else if (line.compare(0, 14, "TUNER_VERSION ") == 0) {
          out.foundTunerVersion = true;
          out.tunerVersion = atoi(line.c_str() + 14);
        }
        else if (line.compare(0, 12, "FINGERPRINT ") == 0) {
          out.foundFingerprint = true;
          out.fingerprint = line.substr(12);
        }
        else if (line.compare(0, 7, "POLICY ") == 0) {
          out.foundPolicy = true;
          out.policy = line.substr(7);
        }
        else if (line.compare(0, 9, "BASELINE ") == 0) {
          out.foundBaseline = true;
          out.baseline = line.substr(9);
        }
        else if (line.compare(0, 12, "ROWSET_HASH ") == 0) {
          out.foundRowSetHash = true;
          out.declaredRowSetHash = line.substr(12);
        }
        else if (line.compare(0, 8, "CREATED ") == 0) {
          out.foundCreated = true;
          out.created = line.substr(8);
        }
        else if (line.compare(0, 6, "MODEL ") == 0) {
          out.foundModel = true;
          out.model = line.substr(6);
        }
        else if (line.compare(0, 9, "MAXBATCH ") == 0) {
          out.foundMaxBatch = true;
          out.maxBatch = atoi(line.c_str() + 9);
        }
        else if (line.compare(0, 6, "BOARD ") == 0) {
          const string v = line.substr(6);
          const size_t x = v.find('x');
          if (x == string::npos) {
            err = "BOARD line " + to_string(lineNo) + ": expected <X>x<Y>, got '" + v + "'";
            return false;
          }
          out.foundBoard = true;
          out.boardX = atoi(v.substr(0, x).c_str());
          out.boardY = atoi(v.substr(x + 1).c_str());
        }
        else if (line.compare(0, 5, "ROWS ") == 0) {
          out.foundDeclaredRows = true;
          out.declaredRows = atol(line.c_str() + 5);
          if (out.declaredRows < 0) {
            err = "ROWS line " + to_string(lineNo) + ": negative row count";
            return false;
          }
        }
        else if (line.compare(0, 9, "NEGATIVE ") == 0) {
          out.negative = true;
          out.negativeReason = line.substr(9);
        }
        else if (line.compare(0, 4, "ROW ") == 0) {
          if (out.rows.size() >= MAX_ROWS_PER_FILE) {
            err = "line " + to_string(lineNo) + ": more than " + to_string(MAX_ROWS_PER_FILE)
                  + " ROW lines (the cap the writer also enforces)";
            return false;
          }
          vk_tune_row_spec row;
          if (!parseRowLine(line.substr(4), lineNo, row, err)) {
            return false;
          }
          out.rows.push_back(row);
        }
        else {
          // An unknown key is refused, not ignored: a file written by a NEWER tuner must
          // fail loudly rather than be read as if the extra field did not matter.
          err = "line " + to_string(lineNo) + ": unknown key '" + line.substr(0, line.find(' ')) + "'";
          return false;
        }
      }
      if (out.foundDeclaredRows && out.declaredRows != (long) out.rows.size()) {
        // This is the truncation guard: a file that lost rows in transit would otherwise
        // be read as a perfectly good, shorter row set.
        err = "declared ROWS " + to_string(out.declaredRows) + " but parsed "
              + to_string(out.rows.size()) + " ROW lines (truncated or hand-edited file?)";
        return false;
      }
      // The ADVISORY set is built from the declared array, not from a hand-written list: the
      // parser cannot drift from REQUIRED_FOR_CACHE, and R12 checks the array's contents.
      for (const char* key : REQUIRED_FOR_CACHE) {
        const string k(key);
        bool found = false;
        if (k == "MODEL") found = out.foundModel;
        else if (k == "BOARD") found = out.foundBoard;
        else if (k == "MAXBATCH") found = out.foundMaxBatch;
        else if (k == "ROWSET_HASH") found = out.foundRowSetHash;
        else if (k == "CREATED") found = out.foundCreated;
        if (!found) {
          out.absentKeys.push_back(k);
        }
      }
      return true;
    }

    // The serialised form.  Kept next to the parser so a field added to one is obviously
    // missing from the other.
    string formatRowsFileImpl(const Decision& d) {
      ostringstream os;
      os << "VERSION " << SCHEMA_VERSION << "\n";
      os << "TUNER_VERSION " << TUNER_VERSION << "\n";
      os << "FINGERPRINT " << d.fingerprint << "\n";
      os << "POLICY " << d.policyFingerprint << "\n";
      os << "BASELINE " << d.baselineVersion << "\n";
      os << "MODEL " << d.modelIdentity << "\n";
      os << "BOARD " << d.nnXLen << "x" << d.nnYLen << "\n";
      os << "MAXBATCH " << d.maxBatchSize << "\n";
      os << "ROWSET_HASH " << rowSetHash(d.rows) << "\n";
      os << "CREATED " << createdAtNow() << "\n";
      // One file = one key, so "this key has rows" and "this key was all-rejected" are
      // mutually exclusive statements.  The writer refuses to emit both; the reader still
      // copes with a hand-edited file containing both (rows win, loudly).  The remaining
      // case -- neither -- is only reachable through EXPORT of an empty row set, which is
      // how the format is demonstrated in tests; the CACHE write path (step 3) must never
      // write it, because loading a file that says nothing would still look like "this key
      // is already decided".
      if (!d.rows.empty() && d.negative) {
        throw StringError("VulkanTuner: refusing to write a file with both ROW and NEGATIVE"
                          " (one file = one key: the two are exclusive statements)");
      }
      // The writer enforces the SAME cap the parser does (MAX_ROWS_PER_FILE).  Refusing to
      // write an over-cap set is the only honest option: writing it would produce a file
      // that the reader then rejects, i.e. a cache that silently never loads.
      if (d.rows.size() > MAX_ROWS_PER_FILE) {
        throw StringError("VulkanTuner: refusing to write more than "
                          + std::to_string(MAX_ROWS_PER_FILE) + " rows");
      }
      os << "ROWS " << d.rows.size() << "\n";
      for (const vk_tune_row_spec& r : d.rows) {
        os << "ROW " << r.path << " " << r.src0_type << " " << r.src1_type
           << " " << r.key_m << " " << r.key_k
           << " " << r.geom.BM << " " << r.geom.BN << " " << r.geom.BK
           << " " << r.geom.WM << " " << r.geom.WN << " " << r.geom.WMITER
           << " " << r.geom.TM << " " << r.geom.TN << " " << r.geom.BS
           << " " << r.geom.pad << " " << r.geom.reshape << " " << r.geom.smaller << "\n";
      }
      return os.str();
    }

    string formatDecisionLine(const Decision& d, int nnXLen, int nnYLen, int maxBatchSize) {
      ostringstream os;
      os << "decision"
         << " device=" << d.deviceName
         << " deviceIdx=" << d.deviceIdx
         << " S=" << nnXLen << "x" << nnYLen
         << " mb=" << maxBatchSize
         << " model=" << d.modelIdentity
         << " policy=" << d.policyFingerprint
         << " baseline=" << d.baselineVersion
         << " rows=" << d.rowCount
         << " source=" << d.rowSource
         << " cache_state=" << d.cacheState
         << " tune_ms=" << d.tuneMs
         << " cache=" << d.cachePath
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
    int maxBatchSize,
    const ModelIdentity& model
  ) {
    const char* importEnv = getenv("KATAGO_VK_TUNE_IMPORT");
    const char* exportEnv = getenv("KATAGO_VK_TUNE_EXPORT");
    const bool enabled = tuneEnabled();

    if (!enabled) {
      // Item 7: TUNE=0 is a complete no-op, EXCEPT that an import/export request must not
      // be dropped in silence -- "I set the env var and nothing happened" is the trap this
      // warning exists for (lesson #46).
      if (importEnv != nullptr && importEnv[0] != '\0') {
        logLine(string("WARNING: KATAGO_VK_TUNE_IMPORT=") + importEnv
                + " was requested but KATAGO_VK_TUNE=0 disables the tuner; nothing was imported"
                  " and no tuned row will be used (reference run stays clean)");
      }
      if (exportEnv != nullptr && exportEnv[0] != '\0') {
        logLine(string("WARNING: KATAGO_VK_TUNE_EXPORT=") + exportEnv
                + " was requested but KATAGO_VK_TUNE=0 disables the tuner; nothing was exported");
      }
      return nullptr;
    }

    const auto t0 = chrono::steady_clock::now();
    shared_ptr<Decision> d = make_shared<Decision>();
    d->deviceIdx = deviceIdx;
    d->nnXLen = nnXLen;
    d->nnYLen = nnYLen;
    d->maxBatchSize = maxBatchSize;
    d->modelIdentity = model.headerValue();
    d->policyFingerprint = policyFingerprint();
    d->baselineVersion = std::to_string(TUNER_POLICY_VERSION);
    d->rowSource = "none";   // stays "none" unless a load actually produced rows

    try {
      char fp[512];
      const size_t fpLen = backendFingerprint(backend, fp, sizeof(fp));
      if (fpLen == 0) {
        logLine("WARNING: the backend could not describe the device; the tuner is disabled for device "
                + to_string(deviceIdx) + " (fail-open: no tuned rows)");
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

      // ONE FILE = ONE KEY: the file name carries the whole outer key, so "re-tune" is a
      // whole-file overwrite rather than a read-modify-write, and a `benchmark -threads
      // 1,2,4` sweep keeps every bucket's rows instead of only the last one's.  The header
      // still holds the same key as the AUTHORITY (a hand-renamed or cross-machine-copied
      // file is caught there); the two cannot drift because both come from this Decision.
      // The model NAME is deliberately absent from the name -- it can contain characters a
      // filesystem dislikes, and it is redundant with the sha for routing.
      d->deviceFile = shortHash(d->fingerprint) + "-" + model.fileTag() + "-"
                    + std::to_string(nnXLen) + "x" + std::to_string(nnYLen)
                    + "-mb" + std::to_string(maxBatchSize) + ".txt";
      d->cacheDir = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkan-tune";
      MakeDir::make(d->cacheDir);
      d->cachePath = d->cacheDir + "/" + d->deviceFile;
      d->cacheFileExists = FileUtils::exists(d->cachePath);
      d->hasPipelineCacheBytes = pipelineCacheBytesBefore(d->pipelineCacheBytesBefore);

      // One loader for both channels: an import is just a cache file that lives somewhere
      // else and is keyed by the operator instead of by the device.  Sharing it is what
      // makes "the file I hand-wrote behaves exactly like the file the tuner wrote" true.
      const char* loadPath = nullptr;
      const char* loadKind = nullptr;
      bool fromImport = false;
      if (importEnv != nullptr && importEnv[0] != '\0') {
        loadPath = importEnv;
        loadKind = "KATAGO_VK_TUNE_IMPORT";
        fromImport = true;
      }
      else if (d->cacheFileExists) {
        loadPath = d->cachePath.c_str();
        loadKind = "cache";
      }

      if (loadPath == nullptr) {
        d->cacheState = "absent";
        d->rowSource = "none";
        d->note = "no cache file for this (device, model, S, mb)";
      }
      else {
        RowsFile file;
        string err;
        string text;
        if (!readWholeFile(loadPath, text, err) || !parseRowsFile(text, file, err)) {
          d->cacheState = "rejected";
          refuseLoad(loadKind, err);
          d->note = string(loadKind) + " unreadable/unparsable, using the stock heuristic";
        }
        else if (!file.foundFingerprint) {
          d->cacheState = "rejected";
          refuseLoad(loadKind, "no FINGERPRINT header, so the device identity is unknown");
          d->note = string(loadKind) + " has no device identity, ignored";
        }
        else if (file.fingerprint != d->fingerprint) {
          // Section 2.1: a row set is accepted only for the SAME device.  Merging across
          // devices is how a tuned row for a 5080 ends up on a 1660.
          d->cacheState = "rejected";
          refuseLoad(loadKind, "fingerprint mismatch (file=" + file.fingerprint
                              + " current=" + d->fingerprint + ")");
          d->note = string(loadKind) + " belongs to another device";
        }
        else if ((file.foundVersion && file.version != SCHEMA_VERSION)
                 || (file.foundTunerVersion && file.tunerVersion != TUNER_VERSION)) {
          // A matching fingerprint is not enough: the same device read by a different
          // tuner may mean something different by the same bytes.
          d->cacheState = "rejected";
          refuseLoad(loadKind, "file VERSION=" + to_string(file.foundVersion ? file.version : -1)
                              + "/TUNER_VERSION=" + to_string(file.foundTunerVersion ? file.tunerVersion : -1)
                              + " but this tuner speaks VERSION=" + to_string(SCHEMA_VERSION)
                              + "/TUNER_VERSION=" + to_string(TUNER_VERSION));
          d->note = string(loadKind) + " was written by another schema version";
        }
        else if (!file.foundPolicy || !file.foundBaseline) {
          // REQUIRED_ANY_CHANNEL: these carry "the policy changed, so this row set is void".
          // Accepting a file without them would make that guarantee vanish silently -- the file
          // would load, the policy would change, and nothing would invalidate anything -- which
          // is why they are refused on BOTH channels rather than tolerated like MODEL/BOARD.
          d->cacheState = "rejected";
          refuseLoad(loadKind, string("missing header key(s) ")
                              + (file.foundPolicy ? "" : "POLICY")
                              + (file.foundPolicy || file.foundBaseline ? "" : ",")
                              + (file.foundBaseline ? "" : "BASELINE")
                              + " (a missing policy key would let a policy change go unnoticed)");
          d->note = string(loadKind) + " does not say which policy its rows were measured under";
        }
        else if (file.policy != d->policyFingerprint) {
          // The row set belongs to a different policy: void, not "possibly still fine".
          d->cacheState = "rejected";
          refuseLoad(loadKind, "policy fingerprint mismatch (file=" + file.policy
                              + " current=" + d->policyFingerprint + ")");
          d->note = string(loadKind) + " was measured under another policy";
        }
        else if (file.baseline != d->baselineVersion) {
          d->cacheState = "rejected";
          refuseLoad(loadKind, "baseline version mismatch (file=" + file.baseline
                              + " current=" + d->baselineVersion + ")");
          d->note = string(loadKind) + " was measured against another policy baseline";
        }
        else if (!fromImport && !file.absentKeys.empty()) {
          // Channel-dependent required keys (2026-09-13 review).  Both channels share this
          // loader, but they differ in what they may tolerate:
          //   cache  -- written by us, so there is NO legitimate reason for a header key to
          //             be missing.  "The name routes, the header verifies" only holds if the
          //             header actually carries the key: without MODEL, a cache copied from
          //             another model and renamed to this key's name would pass every check
          //             left.  That is the one path where the model dimension would silently
          //             stop being verified, so the cache channel requires it.
          //   IMPORT -- the operator names the file explicitly, and hand-written fixtures must
          //             stay possible; absence is tolerated and loudly REPORTED ("[not
          //             key-checked: ...]") rather than only noted internally, so a human can
          //             see that the model dimension was not verified on this channel.
          string missing;
          for (size_t i = 0; i < file.absentKeys.size(); i++) {
            missing += (i ? "," : "") + file.absentKeys[i];
          }
          d->cacheState = "rejected";
          refuseLoad(loadKind, "missing header key(s) " + missing
                              + " (a machine-written cache always carries them;"
                                " KATAGO_VK_TUNE_IMPORT tolerates their absence for hand-written files)");
          d->note = string(loadKind) + " is missing " + missing + ", so the model dimension would not be verified";
        }
        else {
          // Outer key, one item at a time so a refusal names WHICH item did not match.
          string keyMismatch;
          if (file.foundModel && file.model != d->modelIdentity) {
            keyMismatch = "MODEL (file=" + file.model + " current=" + d->modelIdentity + ")";
          }
          else if (file.foundBoard && (file.boardX != nnXLen || file.boardY != nnYLen)) {
            keyMismatch = "BOARD (file=" + to_string(file.boardX) + "x" + to_string(file.boardY)
                        + " current=" + to_string(nnXLen) + "x" + to_string(nnYLen) + ")";
          }
          else if (file.foundMaxBatch && file.maxBatch != maxBatchSize) {
            keyMismatch = "MAXBATCH (file=" + to_string(file.maxBatch)
                        + " current=" + to_string(maxBatchSize) + ")";
          }
          if (!keyMismatch.empty()) {
            d->cacheState = "rejected";
            refuseLoad(loadKind, "outer key mismatch on " + keyMismatch);
            d->note = string(loadKind) + " is for a different model/board/batch";
          }
          else if (!fromImport && file.foundRowSetHash
                   && file.declaredRowSetHash != rowSetHash(file.rows)) {
            // The cache channel: the file must describe the rows it contains.  A mismatch means
            // it was edited or truncated after it was written.
            d->cacheState = "rejected";
            refuseLoad(loadKind, "ROWSET_HASH mismatch (file=" + file.declaredRowSetHash
                                + " computed=" + rowSetHash(file.rows)
                                + "): the file does not describe the rows it contains");
            d->note = string(loadKind) + " was edited after it was written";
          }
          else {
            string keyNote;
            if (!file.absentKeys.empty()) {
              keyNote = " [not key-checked:";
              for (size_t i = 0; i < file.absentKeys.size(); i++) {
                keyNote += (i ? "," : " ") + file.absentKeys[i];
              }
              keyNote += "]";
            }
            if (fromImport && file.foundRowSetHash
                && file.declaredRowSetHash != rowSetHash(file.rows)) {
              // IMPORT: the declared hash is a HINT, never a refusal reason -- a hand-written
              // fixture cannot compute it, and the rows that were actually parsed are what get
              // used.  (The cache channel refuses the same mismatch: see above.)
              keyNote += " [declared ROWSET_HASH differs from the parsed rows;"
                         " the parsed rows are used]";
            }
            d->imported = fromImport;
            if (fromImport) {
              d->importedFrom = loadPath;
            }
            d->rows = file.rows;
            d->rowCount = d->rows.size();
            d->rowSource = fromImport ? "import" : "cache";
            d->cacheState = file.negative ? "negative" : "loaded";
            d->negative = file.negative && file.rows.empty();
            if (file.negative && !file.rows.empty()) {
              // Reachable only from a hand-edited or merged file: the writer refuses to emit
              // both (one file = one key makes them exclusive statements about that key).
              // The rows are the useful thing, so they win -- but saying nothing would turn
              // "we wrote NEGATIVE and it still installed rows" into a mystery (lesson #46).
              d->cacheState = "loaded";
              logLine("WARNING: " + string(loadKind) + " has both ROW and NEGATIVE for this key"
                      " (the writer never emits that); using the " + to_string(d->rowCount)
                      + " row(s) and treating the negative entry as stale");
            }
            if (d->negative) {
              // A negative entry means "this (device, model, S, mb) was DETERMINISTICALLY
              // all-rejected": there is nothing to install and nothing to try.  It is never
              // written on a timeout (t1-schema.md section 1), because "we ran out of time"
              // is not a property of the device.
              d->note = string(loadKind) + ": negative entry (" + file.negativeReason
                       + "), no tuning attempted" + keyNote;
            }
            else {
              d->note = string(loadKind) + " loaded: " + to_string(d->rowCount) + " row(s)"
                      + keyNote;
            }
          }
        }
      }

      // EXPORT writes exactly what a cache file would contain, through the same writer, so
      // the format's round trip is exercised on every run that asks for it (and so a failed
      // write is loud rather than a missing file nobody notices).
      if (exportEnv != nullptr && exportEnv[0] != '\0') {
        const string content = formatRowsFile(*d);
        const string path = exportEnv;
        string err;
        if (writeFileAtomic(path, content, err)) {
          logLine("export path=" + path + " rows=" + to_string(d->rowCount)
                  + " bytes=" + to_string(content.size()) + " ok");
        }
        else {
          logLine("WARNING: export to " + path + " failed: " + err + " (fail-open: nothing changed)");
        }
      }

      d->tuneMs = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
      logLine(formatDecisionLine(*d, nnXLen, nnYLen, maxBatchSize));
      return d;
    }
    catch (const std::exception& e) {
      // Section 3: fail open and loud.  A tuner that can take down a search is worse than
      // no tuner at all.
      logLine("WARNING: the tuner failed for device " + to_string(deviceIdx) + ": " + e.what()
              + " (fail-open: no tuned rows)");
      return nullptr;
    }
  }

  string formatRowsFile(const Decision& decision) {
    return formatRowsFileImpl(decision);
  }

  void installForHandle(const shared_ptr<Decision>& decision, ggml_backend_t backend) {
    if (decision == nullptr) {
      return;
    }
    // The backend prints the per-context install line (it is the only side that can name
    // the context), so this side only has to report a failure.
    const vk_tune_row_spec* rows = decision->rows.empty() ? nullptr : decision->rows.data();
    string err;
    if (!backendInstallRows(backend, rows, decision->rows.size(), err)) {
      logLine("WARNING: install failed for " + decision->deviceName + ": " + err
              + " (fail-open: this context uses the stock heuristic)");
    }
  }

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND
