#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"

#include <chrono>
#include <algorithm>
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

    // The policy baseline counter (t1-schema.md section 4, "baselineVersion").
    //
    // SAME SOURCE as the `p<N>` component of the policy fingerprint: one counter, recorded in
    // two places (the POLICY string and the BASELINE field).  NOT split into a second counter
    // on purpose -- see the note in t1-schema.md section 1.1: two manual counters can each be
    // forgotten, leaving a key that LOOKS maintained, while the diagnostic distinction they
    // would buy ("was this a policy change or a baseline change?") is already given by
    // b<rev16> for anything committed.  Split it later only if the stock baseline becomes
    // mechanically reportable (a backend-reported baseline id), which today would cost the
    // fifth interface name that step 5 wants for probe_candidate.
    //
    // 🔴 BUMP THIS whenever a change alters what an already-tuned row MEANS, without changing
    // any env value and without being caught by the build revision (i.e. an in-place edit of a
    // default, before it is committed):
    //   1. the scalar l/m/s routing thresholds            (ggml_vk_guess_matmul_pipeline)
    //   2. the m<=192 / n<=32 style bucket predicates      (same function)
    //   3. the sentinel set / row-matching rule            (ggml_vk_tuned_row_matches, K1)
    //   4. a fork default that changes a resolved variant  (FA tile default, RMS subgroup
    //                                                       variant, M4 fusion default)
    //   5. the geometry rule table                         (vulkan-tile-validity.h)
    // Committed changes are ALSO covered by b<rev16>; this counter is the belt for the braces,
    // so it must not be the only thing anyone relies on.
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

    // Make a *value* safe to place in a line-oriented text format.  The REAL guarantee, stated
    // exactly: every control character (including newline, 0x0A) becomes '_', so a value can
    // never forge a field or start a new line, and POLICY is the LAST field on its line, so
    // any spaces inside a value are harmless.  It is deliberately NOT "whitespace-safe" in
    // general -- do not reuse it as a filename or CSV sanitiser.
    //
    // NOT used for hashing: hashing a sanitised value would make two different raw values
    // collide (`a\x01b` and `a\x02b` both become `a_b`), which is the opposite of what a
    // fingerprint is for.  See policyFingerprint().
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

    // The request-type switch table (t3-shell-plan.md section 6, "P1").  Every switch that ASKS
    // the tuner to do something must be reported when the tuner is not allowed to act,
    // otherwise the operator gets "I set the env var and nothing happened" (lesson #46).  This
    // was hand-written three times before -- IMPORT, then EXPORT, and RETUNE was silently
    // dropped until a review noticed that the guardrail's own inventory count (2) was the
    // proof.  So the list is DATA and t3-guardrails.py R13 pins ALL FOUR FIELDS: a switch
    // added without thought, or a `vacuousWhen` quietly changed, turns the guardrail red.
    //
    //   env               the switch
    //   vacuousWhen       value that means "the TOOL set it, so under a not-allowed tuner
    //                     nothing was really requested" (nullptr = any set value is a request).
    //                     KATAGO_VK_TUNE_PRIME is set by benchmark/genconfig themselves, so
    //                     warning about "auto" on every reference-arm run would train people to
    //                     ignore warnings -- while the reference arm's probe face is itself an
    //                     acceptance criterion.  A non-default PRIME really was asked for.
    //   ignoredUnderOff   is it ALSO ignored when cfg vulkanTune=off?  IMPORT is a read, so it
    //                     is honoured; EXPORT writes a file the operator named (not the cache),
    //                     so it is honoured too; RETUNE and PRIME ask for TUNING, which off
    //                     forbids.
    //   consequence       what the operator needs to hear.
    struct RequestSwitch {
      const char* env;
      const char* vacuousWhen;
      bool ignoredUnderOff;
      const char* consequence;
    };
    static const RequestSwitch REQUESTS[] = {
      { "KATAGO_VK_TUNE_IMPORT", nullptr, false, "nothing was imported and no tuned row will be used" },
      { "KATAGO_VK_TUNE_EXPORT", nullptr, false, "nothing was exported" },
      { "KATAGO_VK_TUNE_RETUNE", nullptr, true,  "nothing was re-tuned and the cache was not even read" },
      { "KATAGO_VK_TUNE_PRIME",  "auto",  true,  "no prime-time tuning happened (auto is the tool's own setting)" },
    };

    // Report the request switches that cannot be honoured, under whichever reason applies.
    // ONCE PER PROCESS: this is a property of the configuration, not of one context, and a
    // benchmark thread-count sweep creates one context per bucket -- N identical warnings is
    // how a warning becomes wallpaper.
    void reportIgnoredRequests(bool tunerOff, bool readOnly) {
      if (!tunerOff && !readOnly) {
        return;
      }
      static std::atomic<bool> alreadyReported {false};
      bool expected = false;
      if (!alreadyReported.compare_exchange_strong(expected, true)) {
        return;
      }
      const char* why = tunerOff ? "KATAGO_VK_TUNE=0 disables the tuner"
                                 : "vulkanTune=off makes the tuner read-only";
      for (const RequestSwitch & r : REQUESTS) {
        if (readOnly && !tunerOff && !r.ignoredUnderOff) {
          continue;
        }
        const char* v = getenv(r.env);
        if (v == nullptr || v[0] == '\0') {
          continue;
        }
        if (r.vacuousWhen != nullptr && string(v) == r.vacuousWhen) {
          continue;
        }
        logLine(string("WARNING: ") + r.env + "=" + v + " was requested but " + why
                + "; " + r.consequence
                + (tunerOff ? " (reference run stays clean)" : ""));
      }
    }

    // ONE derivation of "this process is in read-only mode".  Today `policy.off` and the write
    // permission are the same fact, but they would otherwise be two SPELLINGS of it, and the
    // moment "read-only" grows a second meaning (e.g. "do not touch an existing cache either")
    // the two would drift apart.
    bool readOnlyMode(const TunePolicy & policy) {
      return policy.off;
    }

    // THE single place that decides whether this process may write a cache file.
    //
    // Why a named predicate rather than `!policy.off` at each site: a guardrail can only check
    // TEXT, so what it must recognise has to be one NAME (the same reason R1/R3 exist).  Three
    // earlier versions of that guardrail were unsound: "is the string policy.off nearby?" is
    // satisfied by an unrelated `if (policy.off) { }`, and "the first `if (` within N lines" is
    // satisfied by an unrelated `if (cacheWriteAllowed(policy)) { log(); }` while the real write
    // sits outside it -- and it went RED for a write whose enclosing condition was simply more
    // than N lines away, which pushes people to write comments that exist only for the guardrail.
    //
    // KATAGO_VK_TUNE=0 is deliberately not consulted here: it never reaches a decision at all.
    bool cacheWriteAllowed(const TunePolicy & policy) {
      return !readOnlyMode(policy);
    }

    // The CLOSED SET of NEGATIVE reasons (t3-shell-plan.md section 6.4 item 2).  A file-level
    // reason is a claim about the DEVICE that outlives this process, so it must be comparable by
    // machine and interpretable by the NEXT release that reads it -- which is exactly why the set
    // is declared data rather than "whatever string the deciding code happens to produce".
    //
    // What may be in here: class (a) only, i.e. the device STRUCTURALLY cannot be tuned (there is
    // no injectable slot, or the row lookup is unreachable by construction).  Class (a) does not
    // depend on any gate / tau / soft-gate parameter, so a policy change can never invalidate it.
    //
    // What must NOT be in here: class (b) (every candidate rejected by the validation gate) and
    // class (c) (timeout / IO failure / DEVICE_LOST / no winner).  (b) is only sound to write once
    // the gate parameters are part of policyFingerprint -- otherwise changing a gate parameter
    // would leave the key permanently skipped -- and (c) is not a property of the device at all
    // (t1-schema.md section 1 discipline (b): "we ran out of time" must never be remembered as
    // "this device cannot be tuned").  Adding a name here is therefore a deliberate act, and both
    // the re-evaluation trigger (section 6.4 item 5) and R17 point at this array.
    //
    // WHY EACH REASON ALSO DECLARES ITS INVALIDATION RULE (2026-09-13 review): a reason alone is
    // not enough.  "This device cannot be tuned" is a claim that OUTLIVES the process, and a claim
    // is only sound if something is guaranteed to void it -- otherwise the file becomes "this key
    // is permanently skipped", which is the worst outcome the negative cache can produce.  So the
    // rule is DATA here (pinned by R17), not a sentence in a comment.
    //
    //   `full-key-set` = the ONLY rule implemented today, and the strongest one available: the
    //   loader refuses the file unless FINGERPRINT, TUNER_VERSION, POLICY, BASELINE and (on the
    //   cache channel) MODEL, BOARD, MAXBATCH and ROWSET_HASH all match, so a change of device,
    //   driver, capability, tuner code, policy, model, board or batch voids the claim.  Every
    //   reason therefore has the SHORTEST possible lifetime, and over-invalidation is the safe
    //   direction (the cost is one re-decision, not a stale skip).
    //
    // A reason that needs anything else is a FORMAT-LEVEL act, not an edit here: a SHORTER
    // lifetime means a new invalidation condition, which means a new key in the header and the
    // file name (R12 forces that decision too); a LONGER lifetime -- surviving a key change -- is
    // not expressible at all, because the file name carries the whole key.  The practical reason
    // the rule is per-reason: the reasons are not obviously equal (no-injection-slot is a hard
    // fact about the device, no-entry leans on the model/family), so "they happen to share a rule
    // today" has to be written down instead of being assumed.
    //
    // WHERE `full-key-set` IS REDEEMED (a declared rule that nothing redeems is a comment in
    // disguise, and this struct carries no runtime weight at all): the LOADER's key comparisons --
    // makeDecision()'s refusal chain (fingerprint, VERSION/TUNER_VERSION, POLICY, BASELINE, then
    // the outer key item by item, then ROWSET_HASH on the cache channel) together with
    // parseRowsFile()'s required-key loop, which is fed by REQUIRED_ANY_CHANNEL/REQUIRED_FOR_CACHE.
    // Nothing mechanically ties the value to those lines: R17 pins the value AND pins that each
    // named comparison still exists somewhere in this file, and the behavioural half is the
    // smoke's refusal legs (S6, A5/A6, A14/A15/A16, A18, A31/A32).  Change either side and both
    // have to be revisited.
    //
    // ⚠️ KNOWN GAP, deliberately left visible (2026-09-13, awaiting a ruling): the comparisons
    // above are written as `file.foundVersion && file.version != SCHEMA_VERSION`, so a file that
    // simply OMITS VERSION / TUNER_VERSION is neither refused nor REPORTED -- verified by hand:
    // such a file loads with no `[not key-checked: ...]` note on either channel, while
    // t1-schema.md section 1 lists both as required.  Adding them to REQUIRED_FOR_CACHE would fix
    // both halves at once (cache refuses absence, IMPORT notes it) and is the recommended fix.
    struct NegativeReasonSpec {
      const char* reason;
      const char* invalidation;
    };
    const NegativeReasonSpec NEGATIVE_REASONS[] = {
      { "no-injection-slot", "full-key-set" },
      { "no-entry",          "full-key-set" },
    };

    // The closed set holds on BOTH sides: the parser (so an uninterpretable file is refused
    // instead of loading as if it had said nothing) and the cache writer's "is there anything to
    // write?" predicate below (so free text can never reach the disk).  A set only the reader
    // checks is a set on paper.
    bool negativeReasonKnown(const string & reason) {
      for (const NegativeReasonSpec & r : NEGATIVE_REASONS) {
        if (reason == r.reason) {
          return true;
        }
      }
      return false;
    }

    // "Is there anything to write?" -- the SECOND half of the write permission.
    // `cacheWriteAllowed(policy)` answers "may this process touch the cache at all" (the user's
    // mode); this answers "does this decision say anything" (the decision's state), and they are
    // separate because they fail for different reasons and are fixed by different people.  The
    // state it rejects is real: a file carrying neither ROW nor NEGATIVE loads without complaint
    // and reads as "this key is already decided" (see the writer comment in formatRowsFileImpl),
    // so the CACHE path must refuse to create one while EXPORT may still write it (that is how
    // the format is demonstrated in tests).
    bool cacheEntryWritable(const Decision & d) {
      if (!d.rows.empty()) {
        return true;
      }
      return d.negative && negativeReasonKnown(d.negativeReason);
    }

    // The CLOSED SET of decision outcomes and whether each one may be REMEMBERED (step "3-4").
    //
    // Why a table and not a bool: the rule is "a measurement-level failure is never written", and
    // the failure modes are not interchangeable -- a timeout, an IO error, a lost device and a run
    // with no winner all mean "this attempt did not produce an answer", while a structural
    // all-reject means "there is no answer to produce" and a completed measurement means "here is
    // the answer".  Only those last two may be written, and they are different KINDS of writable
    // (a row set vs a NEGATIVE entry), which is why the class is declared as well -- the class is
    // what section 6.4's rule table names, and what step 4/5 will branch on.
    //
    // Class (b) (every candidate rejected by the validation gate) is declared NOT writable and
    // stays that way until the gate / tau / soft-gate parameters enter policyFingerprint, or
    // changing a gate parameter would leave the key permanently skipped (section 6.4 items 1/5 --
    // the same argument R17 makes for the file's reason).
    //
    // The lookup is FAIL-CLOSED (see outcomeWritable): an outcome this build does not know is not
    // writable, so a caller that invents a state cannot write by accident.
    struct DecisionOutcomeSpec {
      const char* outcome;
      // `klass` has exactly three roles, and NONE of them is the runtime verdict:
      //   (1) R19's whitelist key in the guardrail (t3-guardrails.py WRITABLE_CLASSES);
      //   (2) the self-test's FIXTURE SELECTOR (:1084/:1092 choose which shape to build);
      //   (3) a documentation label on the outcome name.
      // The production decision path never reads it: `outcomeWritable` uses `writable` + `condition`
      // only.  A rename is therefore LOUD twice over, never silent -- R19 reddens (whitelist), and
      // the self-test reddens too (a class with no fixture builder makes `outcomeWritable` disagree
      // with `o.writable`), which is what makes "add a writable class, forget its fixture"
      // impossible to land green.
      const char* klass;       // "structural" / "completed" / "gate" / "measurement"
      bool writable;
      // The EXTRA condition on top of `writable` (G-D20): "" = none, "device-only" = every
      // rejection reason must be a device property.  It lives in the table on purpose -- a
      // condition expressed as a branch on `klass` would let a class rename silently change what
      // may be written, which is the "implicit knowledge in a branch" the first design rule bans.
      const char* condition;
    };
    const DecisionOutcomeSpec DECISION_OUTCOMES[] = {
      { "structural-all-reject", "structural",  true,  "device-only" },
      { "measured",              "completed",   true,  ""            },
      { "gate-reject",           "gate",        false, ""            },
      { "running",               "measurement", false, ""            },
      { "timeout",               "measurement", false, ""            },
      { "io-error",              "measurement", false, ""            },
      { "device-lost",           "measurement", false, ""            },
      { "no-winner",             "measurement", false, ""            },
    };

    // The extra condition named by a table row (G-D20).  Fail-closed: an unknown condition is
    // never satisfied, so a typo in the table cannot make something writable.
    bool outcomeConditionOk(const Decision & d, const char * condition) {
      const string c = (condition == nullptr) ? string() : string(condition);
      if (c.empty()) {
        return true;
      }
      if (c == "device-only") {
        return d.allRejectReasonsAreDevice;
      }
      return false;
    }

    bool outcomeWritable(const Decision & d) {
      for (const DecisionOutcomeSpec & o : DECISION_OUTCOMES) {
        if (d.outcome == o.outcome) {
          return o.writable && outcomeConditionOk(d, o.condition);
        }
      }
      return false;
    }

    // The probe face belongs to the backend (KATAGO_VK_T2_PROBE, bit 8 = the tuned-row surface,
    // which includes its validator self-test).  The shell reads the SAME variable with the SAME
    // grammar -- `all` or a base-10 bitmask, see vk_t2_probe's comment -- so one env asks both
    // sides for their self-checks.  Never reached under KATAGO_VK_TUNE=0: makeDecision returns
    // before this (and the backend suppresses its probes there for the same reason).
    bool t2ProbeBit8() {
      const char* v = getenv("KATAGO_VK_T2_PROBE");
      if (v == nullptr || v[0] == '\0') {
        return false;
      }
      const unsigned long bits = (strcmp(v, "all") == 0) ? 0xFul : strtoul(v, nullptr, 10);
      return (bits & 8ul) != 0;
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
      // NOTE: the build revision is truncated to 16 characters and that is NOT the lossy kind
      // of truncation the knob values below must avoid -- a SHA is a fixed-length prefix-free
      // identifier, so no two revisions share their first 16 hex characters in practice, and
      // keeping it readable in the log is worth more than the extra 48 characters.  Do not
      // "fix" it with the hash rule below: the revision must stay greppable.
      string s = "b" + sanitizeValue(string(GIT_REVISION), 16)
               + "|p" + std::to_string(TUNER_POLICY_VERSION);
      for (const char* knob : KNOBS) {
        const char* v = getenv(knob);
        s += "|";
        s += knob;
        s += "=";
        if (v == nullptr || v[0] == '\0') {
          s += "-";
        }
        else {
          // SHORT VALUES AS-IS, LONG VALUES HASHED -- and the hash is taken over the RAW value.
          // Two earlier mistakes are fixed here: truncating (lossy AND silent: two
          // KATAGO_VK_MM_TILE=custom:... values differing past the cap produced the same
          // fingerprint, so a policy change would have reused the cache), and hashing the
          // SANITISED value (`a\x01b` and `a\x02b` both sanitise to `a_b`, i.e. two different
          // policy values would collide).  The length recorded is the RAW length, for the same
          // reason.  shortHash() output is hex, so it needs no sanitising to stay line-safe.
          const size_t MAX_VALUE = 24;
          const string raw(v);
          if (raw.size() <= MAX_VALUE) {
            // Short enough to read: keep it, but make it line-safe.
            s += sanitizeValue(raw, MAX_VALUE);
          }
          else {
            s += "#" + shortHash(raw) + "/" + std::to_string(raw.size());
          }
        }
      }
      return s;
    }

    // FNV-1a over the rows' identity+geometry, spelled out field by field (never over the
    // struct's bytes: padding would make the hash depend on the compiler).  This is what
    // ROWSET_HASH declares, and on the cache channel a mismatch means the file does not
    // describe the rows it contains.
    //
    // FNV-1a 64 with the STANDARD parameters: offset basis 0xCBF29CE484222325 = 14695981039346656037,
    // prime 0x100000001B3 = 1099511628211.  They MUST be the standard values (t1-schema.md 1.3):
    // ROWSET_HASH is a cross-repo contract, so a port that implements FNV-1a from the standard has
    // to agree with us byte for byte.  The empty row set (the four zero bytes of the row count) is
    // pinned by cacheWriteSelfTest() below -- that is the leg that keeps this constant honest.
    string rowSetHash(const vector<vk_tune_row_spec>& rows) {
      uint64_t h = 14695981039346656037ull;
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

    // Read the whole file rather than using FileUtils::readFile: that helper throws on failure
    // and this path must return an error string instead (every load failure here is fail-open +
    // loud, never an exception out of the hook).  Do not "simplify" this back to readFile().
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
    // `deviceSubgroupSize` comes from Decision.deviceSubgroupSize (injected by the backend through
    // the fingerprint call).  It is a PARAMETER, not a global: one process can hold several devices,
    // and a global would let one device's wave judge another device's rows (G-C3's second pitfall).
    bool parseRowLine(const string& rest, size_t lineNo, uint32_t deviceSubgroupSize, vk_tune_row_spec& out, string& err) {
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
      // The geometry table needs the wave: BS is derived as num_warps * wave, so whether a row is
      // legal depends on the device it is FOR.  Keeping the clause here (instead of dropping it at
      // parse time) is what preserves the SSOT property -- one bad row, one identical reason string
      // -- for the devices the rows are meant for.
      // R1b-2b close-out (2026-09-17): the CONSUMER guards the wave, not the caller.  Why here and
      // not only at the Decision: `vk_tile_wave_for(0)` returns 8 (it must stay identical to stock's
      // `subgroup_size_8 = max(sg,8)`, so it can never be changed to return 0), which means an
      // unfilled subgroup size would otherwise be judged as wave 8 -- silently legal-or-not for the
      // wrong hardware.  The predicate's own `wave == 0` clause stays as the second line of defence;
      // this one is what a future caller (the ⑤ enumerator) cannot forget, because the value is
      // consumed here.
      if(deviceSubgroupSize == 0) {
        err = "ROW line " + to_string(lineNo) + ": the device's subgroup/wave size is unknown (0), "
              "so no geometry can be judged (a 0 here means the caller never filled it in)";
        return false;
      }
      const char* reason = vk_tile_candidate_reason(path, out.geom, vk_tile_wave_for(deviceSubgroupSize));
      if (reason != nullptr) {
        err = "ROW line " + to_string(lineNo) + ": illegal geometry: " + reason;
        return false;
      }
      return true;
    }

    bool parseRowsFile(const string& text, uint32_t deviceSubgroupSize, RowsFile& out, string& err) {
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
          // Same rule as an unknown key (see the final else): a claim this tuner cannot interpret
          // must not be read as if it had said nothing.  Be precise about WHAT the reason decides
          // (this comment used to overclaim): every reason currently declares the SAME invalidation
          // rule (the whole key set -- see NEGATIVE_REASONS), so the reason does not change how
          // long the claim lives TODAY.  What it changes is what the claim MEANS, and that is
          // enough to refuse a string we do not know: "this device structurally cannot be tuned"
          // and an arbitrary sentence are not the same assertion, and only one of them is safe to
          // act on.  Declaring the rule per reason is what keeps a FUTURE reason with a different
          // lifetime from being added without saying so (R17 pins both fields).
          const string reason = line.substr(9);
          if (!negativeReasonKnown(reason)) {
            err = "NEGATIVE line " + to_string(lineNo) + ": unknown reason '" + reason
                + "' (the reason is a closed set; see NEGATIVE_REASONS)";
            return false;
          }
          out.negative = true;
          out.negativeReason = reason;
        }
        else if (line.compare(0, 4, "ROW ") == 0) {
          if (out.rows.size() >= MAX_ROWS_PER_FILE) {
            err = "line " + to_string(lineNo) + ": more than " + to_string(MAX_ROWS_PER_FILE)
                  + " ROW lines (the cap the writer also enforces)";
            return false;
          }
          vk_tune_row_spec row;
          if (!parseRowLine(line.substr(4), lineNo, deviceSubgroupSize, row, err)) {
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
      // how the format is demonstrated in tests; the CACHE write path cannot create one
      // because cacheEntryWritable() refuses it before this function is called.
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
      if (d.negative) {
        // The NEGATIVE segment.  Without this the negative channel had no WRITER at all: a
        // decision that was (or had loaded) a negative entry exported as `ROWS 0` and nothing
        // else -- i.e. the marker was silently dropped and the round trip was broken, which is
        // the same "it looked fine" failure the read side already refuses to make.
        //
        // The reason is taken from the Decision (never a literal here, so R17 can require that)
        // and re-checked against the closed set: the parser refuses an unknown reason, so a file
        // this writer produces is always re-readable.  `d.rows` is empty whenever `d.negative` is
        // set (the guard above throws otherwise), which is also why the ROWSET_HASH written above
        // is the hash of the EMPTY row set -- a deterministic value, not a placeholder.
        if (!negativeReasonKnown(d.negativeReason)) {
          throw StringError("VulkanTuner: refusing to write a NEGATIVE entry with reason '"
                            + d.negativeReason + "' (it is not in the closed set, and a file this "
                            "tuner cannot re-read must not be written)");
        }
        os << "NEGATIVE " << d.negativeReason << "\n";
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
         << " tuneMode=" << d.tuneModeText
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

  size_t backendFingerprint(ggml_backend_t backend, char* out, size_t outSize, uint32_t* sgOut) {
    return ::ggml_backend_vk_tune_fingerprint(backend, out, outSize, sgOut);
  }

  bool backendInstallRows(ggml_backend_t backend, const vk_tune_row_spec* rows, size_t nRows, string& err,
                          vk_tune_install_result& out) {
    vector<char> errBuf(512, '\0');
    const bool ok = ::ggml_backend_vk_tune_install_rows(backend, rows, nRows, errBuf.data(), errBuf.size(), &out);
    if (!ok) {
      err = string(errBuf.data());
    }
    return ok;
  }

  // THE single cache write path (t3-shell-plan.md section 9.1 R16, step "3-3").
  //
  // Why a named function rather than an `if` at the call site: the write permission has to be one
  // recognisable NAME for the guardrail to check, and the check is STRUCTURAL -- this body is the
  // only place a cache writeFileAtomic call may sit, and both permission predicates have to be
  // named in it BEFORE the write.  It is deliberately NOT declared in vulkantuner.h: the header
  // says what the caller (vulkanbackend.cpp) needs, and this is an internal step of the shell.
  //
  // NO CALLER YET, and that is the documented state of step 3-3 rather than an oversight.  The
  // first class-(a) verdict ("this device structurally cannot be tuned") can only come from the
  // enumerator, which is step 5, so at this cut every runnable path ends with "nothing to say" and
  // MUST NOT write.  The consequence is recorded instead of hidden (t1-schema.md section 1): step
  // 3 lands the function and the rules, step 4 lands the end-to-end, and the POSITIVE leg for this
  // write path (a synthetic class-(a) all-reject producing a cache file whose header matches the
  // schema) is step 4's acceptance item.  What this cut CAN check is the negative side: R16's
  // structure, plus the behaviour legs that plant a cache and require it byte-identical and that
  // run with no cache file at all and require none to appear.
  //
  // The CALLER logs; this function only reports -- and that is now TRUE rather than aspirational:
  // formatRowsFile() REFUSES some decisions by throwing (both ROW and NEGATIVE in one file; a
  // reason outside the closed set), so the conversion to an error string happens HERE, once.  The
  // read path states the same contract for the same reason ("every load failure is fail-open +
  // loud, never an exception out of the hook"), and a caller that later moves outside the
  // decision's try/catch must not be able to turn a cache refusal into a crash.  Keeping the
  // function silent is also what keeps one writer per line (section 9.3-9): the message is born
  // with the path that can actually produce it, which is the caller (step 4).
  bool writeCacheIfAllowed(const Decision& d, const TunePolicy& policy, string& err) {
    if (!cacheWriteAllowed(policy)) {
      // Read-only (cfg vulkanTune=off, or any future second meaning of "read-only"): the cache may
      // be READ but never written.  Reaching here is a bug in the caller, so it is refused loudly
      // rather than silently skipped.
      err = "the cache is read-only in this mode";
      return false;
    }
    if (!outcomeWritable(d)) {
      // THE state gate (step 3-4).  "We ran out of time" must never be remembered as "this device
      // cannot be tuned", and neither must a lost device, an IO failure or a run with no winner:
      // those describe THIS attempt, and a cache file describes the DEVICE.  Only a completed
      // measurement or a structural all-reject gets past here (see DECISION_OUTCOMES), and an
      // outcome nobody declared is refused too (fail-closed).
      err = "the decision is not in a writable state ('" + d.outcome + "'): only a completed "
            "measurement or a structural all-reject may be remembered";
      return false;
    }
    if (!cacheEntryWritable(d)) {
      err = "nothing to write: no rows and no negative entry (a file that says neither would "
            "load as 'this key is already decided')";
      return false;
    }
    // WRITE-PATH: cache
    try {
      return writeFileAtomic(d.cachePath, formatRowsFile(d), err);
    }
    catch (const std::exception& e) {
      err = string("the cache entry was refused: ") + e.what();
      return false;
    }
  }

  // Exercise the outcome table and the write gates THROUGH the real function, once per process,
  // under probe bit 8.
  //
  // Why this exists (2026-09-13 review of step 3-4): "a measurement-level failure is never
  // written" has no non-vacuous executor at this step -- the cache write path has no caller and
  // there is no measurement loop yet, so an end-to-end leg asserting "no file was created" would
  // pass because nothing ever ATTEMPTS a write, not because the rule works (the vacuous judgement
  // section 9.2 keeps catching).  This calls the real writer with a real Decision for every
  // declared outcome and requires exactly the declared verdict:
  //   * an unwritable outcome must be refused by the write path itself, with a write-allowed
  //     policy (so the refusal comes from the state gate, not from the mode);
  //   * a writable outcome must pass the STATE gate and be stopped only by the read-only policy --
  //     the positive direction is therefore checked without ever writing a file that no real
  //     verdict produced (the real positive leg belongs to step 4).
  // Then it requires that nothing appeared at the probe path.  The four E2E legs that cannot be
  // written yet (timeout / io-error / device-lost / no-winner) are registered for step 5 WITH a
  // mandatory precondition: prove the failure happened first, then assert the file was not created.
  void cacheWriteSelfTest(const TunePolicy & policy, const string & cacheDir) {
    TunePolicy readOnly = policy;
    readOnly.off = true;
    const string probePath = cacheDir + "/self-test-must-not-exist.txt";
    int declared = 0, refused = 0, allowed = 0;
    bool ok = true;
    string firstBad;
    // R0 legs: pin rowSetHash() to the STANDARD FNV-1a 64 parameters AND to the wire protocol
    // (field set, field order, 4-byte little-endian feed).  Both expected values come from an
    // INDEPENDENT implementation (python, logs-t3/1b2b/r0-*), never from this function:
    //   empty row set (the four zero bytes of the row count) -> basis * prime^4 mod 2^64
    //                                                        = 0x4d25767f9dce13f5
    //   one legal row (count + 17 fields = 18 feeds)          = 0xe3cde8d1133b7112
    // The empty vector alone does NOT pin the protocol: four zero bytes leave the digest
    // insensitive to the xor order, the byte order and the field order.  Sensitivity proof --
    // both values DIFFER from the expected one, so a weakened vector cannot pass by accident:
    //   swap two field positions -> 0x12b27bd83877cd52 ;  feed the bytes big-endian -> 0x85f45fe6df78624c
    // LITERAL DISCIPLINE (t1-schema.md 1.3): these two literals may only come from the
    // independent implementation.  REFERENCE them; never recompute them in place and never edit
    // them to match a red leg -- "make the literal agree with the code" is the ONE way this
    // check can silently become a no-op.
    {
      vk_tune_row_spec v {};
      v.path = VK_TILE_PATH_SCALAR;
      v.src0_type = 1;   // f16
      v.src1_type = 1;   // f16
      v.key_m = 192;
      v.key_k = 512;
      v.geom.BM = 128; v.geom.BN = 128; v.geom.BK = 32;
      v.geom.WM = 32;  v.geom.WN = 64;  v.geom.WMITER = 2;
      v.geom.TM = 2;   v.geom.TN = 2;   v.geom.BS = 256;
      v.geom.pad = 1;  v.geom.reshape = 0; v.geom.smaller = 0;
      // The vector's row must be LEGAL, and that is checked here rather than claimed: a vector
      // over an illegal row would still hash, but it would stop describing a row the tuner can
      // actually produce (and if the predicate ever started rejecting it, nobody would notice).
      // R1b-2b: the wave is an INPUT now, so this check must exercise BOTH generations -- and it
      // SYNTHESIZES them rather than reading the injected value: this function has no device (it runs
      // from the cache-dir self-test) and the wave-dependent clauses still need a writer on both
      // sides.  `v` is the wave32 fixture (BM=128,BN=128,WM=32,WN=64,WMITER=2,TM=2,TN=2,BS=256 =>
      // num_warps = 4*2 = 8 = 256/32), so wave64 must REFUSE it; `v64` is the mirror
      // (WM=WN=64,WMITER=4,TM=TN=4 => num_warps = 2*2 = 4 = 256/64), so wave32 must refuse it.
      // The cross-check is the point: a fixture that only ever passes under one wave would also pass
      // if the parameter were ignored entirely, so it would prove nothing about the wave.
      const uint32_t wave32 = vk_tile_wave_for(32);
      const uint32_t wave64 = vk_tile_wave_for(64);
      const char * why     = vk_tile_candidate_reason(v.path, v.geom, wave32);              // want legal
      const char * why64no = vk_tile_candidate_reason(v.path, v.geom, wave64);              // want refused
      vk_tune_row_spec v64 = v;
      v64.geom.WM = 64; v64.geom.WN = 64; v64.geom.WMITER = 4; v64.geom.TM = 4; v64.geom.TN = 4;
      v64.geom.BS = 256;
      const char * why64ok = vk_tile_candidate_reason(v64.path, v64.geom, wave64);          // want legal
      const char * why32no = vk_tile_candidate_reason(v64.path, v64.geom, wave32);          // want refused
      const string gotEmpty = rowSetHash(vector<vk_tune_row_spec>());
      const string gotOne   = rowSetHash(vector<vk_tune_row_spec>{ v });
      const string wantEmpty = "4d25767f9dce13f5";
      const string wantOne   = "e3cde8d1133b7112";
      if (why != nullptr || why64no == nullptr || why64ok != nullptr || why32no == nullptr
          || gotEmpty != wantEmpty || gotOne != wantOne) {
        ok = false;
        firstBad = "rowSetHash vector: legal=" + string(why == nullptr ? "yes" : why)
                   + " empty=" + gotEmpty + " (want " + wantEmpty + ")"
                   + " one-row=" + gotOne + " (want " + wantOne + ")"
                   + " wave cross-check: v@32=" + (why == nullptr ? "legal" : "REFUSED")
                   + " v@64=" + (why64no == nullptr ? "LEGAL (BUG)" : "refused")
                   + " v64@64=" + (why64ok == nullptr ? "legal" : "REFUSED (BUG)")
                   + " v64@32=" + (why32no == nullptr ? "LEGAL (BUG)" : "refused");
      }
    }
    for (const DecisionOutcomeSpec & o : DECISION_OUTCOMES) {
      Decision d;
      d.outcome = o.outcome;
      d.cachePath = probePath;
      d.nnXLen = 19;
      d.nnYLen = 19;
      d.maxBatchSize = 8;
      d.fingerprint = "self-test";
      d.policyFingerprint = "self-test";
      d.baselineVersion = "1";
      d.modelIdentity = "self-test";
      if (o.writable && string(o.klass) == "structural") {
        // The one writable shape that says something without rows.
        d.negative = true;
        d.negativeReason = "no-injection-slot";
        // G-D20: "the device has no injectable slot" is writable only when every rejection reason
        // is a device property; the fixture has to say so explicitly (unset => not writable).
        d.allRejectReasonsAreDevice = true;
      }
      else if (o.writable && string(o.klass) == "completed") {
        // The rows-file shape: one legal row, so the CONTENT gate passes and only the state gate
        // is left to check.  The geometry is the one the smoke's A3 leg proves legal, and the
        // verdict below re-checks it against the rule table -- a self-test whose own fixture went
        // illegal would otherwise test nothing while looking green.
        vk_tune_row_spec r {};
        r.path = VK_TILE_PATH_SCALAR;
        r.src0_type = 1;
        r.src1_type = 0;
        r.key_m = 0;
        r.key_k = 0;
        r.geom.BM = 64;
        r.geom.BN = 64;
        r.geom.BK = 0;
        r.geom.WM = 32;
        r.geom.WN = 32;
        r.geom.WMITER = 1;
        r.geom.TM = 4;
        r.geom.TN = 4;
        r.geom.BS = 128;
        r.geom.pad = -1;
        r.geom.reshape = -1;
        r.geom.smaller = -1;
        d.rows.push_back(r);
      }
      string err;
      bool verdict = false;
      const bool stateOk = (outcomeWritable(d) == o.writable);   // the table agrees with itself
      // The condition's OTHER direction, per row (G-D20): a row that names an extra condition must
      // be REFUSED when that condition does not hold.  Asserting only the satisfying direction would
      // leave the condition itself untested (and would let it decay into a comment).
      if (string(o.condition).size() > 0) {
        Decision dCond = d;
        dCond.allRejectReasonsAreDevice = false;
        if (outcomeWritable(dCond)) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("outcome ") + o.outcome + " is writable while its condition ("
                       + o.condition + ") does not hold";
          }
        }
      }
      // The `completed` fixture builds one legal row (BM=BN=64, WM=WN=32 => num_warps = 4, BS=128),
      // which is a WAVE32 row: synthesized here for the same reason as above (no device in this
      // function), and the writer of that leg documents the wave it assumes.
      const bool fixtureOk = d.rows.empty()
                          || vk_tile_candidate_reason(d.rows[0].path, d.rows[0].geom,
                                                      vk_tile_wave_for(32)) == nullptr;
      if (!o.writable) {
        // The STATE gate itself must refuse, and the error string must name it -- with a
        // write-ALLOWED policy, so the mode cannot be the thing that refused, and with the string
        // checked so the CONTENT gate (which would also refuse `gate-reject`) cannot be mistaken
        // for the state gate.
        const bool wrote = writeCacheIfAllowed(d, policy, err);
        verdict = !wrote && err.find("not in a writable state") != string::npos;
        refused++;
      }
      else {
        // A writable outcome must get PAST the state gate: with a read-only policy the refusal has
        // to come from the permission gate, which its message names.  Written this way the
        // positive direction is checked without ever writing a file that no real verdict produced
        // (the first version of this test called the writer with a write-allowed policy here and
        // dutifully created the probe file -- found by the test itself, which is the point).
        string err2;
        const bool wroteReadOnly = writeCacheIfAllowed(d, readOnly, err2);
        verdict = !wroteReadOnly && err2.find("read-only") != string::npos
               && cacheEntryWritable(d);
        allowed++;
      }
      declared++;
      if (!verdict || !stateOk || !fixtureOk) {
        ok = false;
        if (firstBad.empty()) {
          firstBad = string(o.outcome)
                   + (!fixtureOk ? " has an ILLEGAL self-test fixture" : " was not refused as declared");
        }
      }
    }
    // G-D20 leg (R1b-2a): a `structural-all-reject` whose reasons came from OUR implementation
    // gap must be refused by the state gate even though the declared table says the outcome is
    // writable.  This is the RULE's execution body, and it needs no write side: the plumbing that
    // fills `allRejectReasonsAreDevice` from the install accounting is what waits for B5/B6.
    {
      Decision d;
      d.cachePath = probePath;
      d.nnXLen = 19;
      d.nnYLen = 19;
      d.maxBatchSize = 8;
      d.fingerprint = "self-test";
      d.policyFingerprint = "self-test";
      d.baselineVersion = "1";
      d.modelIdentity = "self-test";
      d.outcome = "structural-all-reject";
      d.negative = true;
      d.negativeReason = "no-injection-slot";
      d.allRejectReasonsAreDevice = false;   // one reason came from our own gap
      const bool refusedOurGap = !outcomeWritable(d);
      d.allRejectReasonsAreDevice = true;    // every reason is a device property
      const bool allowedDeviceOnly = outcomeWritable(d);
      if (!refusedOurGap || !allowedDeviceOnly) {
        ok = false;
        if (firstBad.empty()) {
          firstBad = string("G-D20: our-gap structural-all-reject writable=")
                     + (refusedOurGap ? "no" : "YES (BUG)") + ", device-only writable="
                     + (allowedDeviceOnly ? "yes" : "NO (BUG)");
        }
      }
    }

    // R1b-2b close-out: the two ways a wave-less call can reach the geometry table, asserted here
    // because both need NO device -- the earlier registration said "implemented but unreachable on
    // this machine", which was the wrong diagnosis: vk_tile_wave_for(0) is 8, so the predicate's
    // `wave == 0` clause is unreachable through a caller that uses the helper, and what actually
    // protects us is the consumer guard above.  A self-test can reach both directly.
    {
      vk_tile_geometry g0 = {};
      g0.BM = 128; g0.BN = 128; g0.BK = 0;
      g0.WM = 32;  g0.WN = 64;  g0.WMITER = 2;
      g0.TM = 2;   g0.TN = 2;   g0.BS = 256;
      g0.pad = -1; g0.reshape = -1; g0.smaller = -1;
      const char* r0 = vk_tile_candidate_reason(VK_TILE_PATH_SCALAR, g0, 0);
      if(r0 == nullptr || string(r0).find("wave == 0") == string::npos) {
        ok = false;
        if(firstBad.empty())
          firstBad = string("predicate with wave=0 said ") + (r0 == nullptr ? "LEGAL (BUG)" : r0);
      }
      RowsFile rf0; string err0;
      const bool parsed0 = parseRowsFile("ROW scalar 1 1 0 0 32 32 32 0 0 0 0 0 0 0 0 0\n", 0, rf0, err0);
      if(parsed0 || !rf0.rows.empty()) {
        ok = false;
        if(firstBad.empty())
          firstBad = "parseRowsFile accepted a ROW line with an unknown wave (0): rows="
                     + to_string(rf0.rows.size());
      }
      else if(err0.find("subgroup") == string::npos && err0.find("unknown") == string::npos) {
        ok = false;
        if(firstBad.empty())
          firstBad = "the unknown-wave ROW refusal does not name the cause: " + err0;
      }
    }

    const bool absent = !FileUtils::exists(probePath);    logLine("cache write self-test: " + to_string(declared) + " outcome(s) declared, "
            + to_string(refused) + " refused by the state gate, " + to_string(allowed)
            + " state-writable; probe file " + (absent ? "absent" : "PRESENT") + " ["
            + ((ok && absent) ? "ok" : "FAIL") + "]"
            + (ok ? "" : " first problem: " + firstBad));
  }

  shared_ptr<Decision> makeDecision(
    ggml_backend_t backend,
    int deviceIdx,
    const string& homeDataDirOverride,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    const ModelIdentity& model,
    const TunePolicy& policy
  ) {
    const char* importEnv = getenv("KATAGO_VK_TUNE_IMPORT");
    const char* exportEnv = getenv("KATAGO_VK_TUNE_EXPORT");
    // KATAGO_VK_TUNE_RETUNE=1: ignore whatever is on disk (rows AND the negative entry) and
    // decide again.  It is the operator's "I do not trust this file" switch, and it is
    // deliberately about the whole file rather than about the rows or the negative marker:
    // one file = one key, so a partial override would leave a half-trusted state.
    const bool retune = getenv("KATAGO_VK_TUNE_RETUNE") != nullptr
                     && getenv("KATAGO_VK_TUNE_RETUNE")[0] != '\0'
                     && !(getenv("KATAGO_VK_TUNE_RETUNE")[0] == '0'
                          && getenv("KATAGO_VK_TUNE_RETUNE")[1] == '\0');
    const bool enabled = tuneEnabled();

    if (!enabled) {
      reportIgnoredRequests(true, false);
      return nullptr;
    }
    // cfg vulkanTune=off: read-only.  Still LOUD about the requests that cannot be honoured
    // (RETUNE, PRIME): "I set RETUNE and nothing happened" is the same trap as under the master
    // switch, and this mode is the one a user is most likely to combine with a request.
    reportIgnoredRequests(false, readOnlyMode(policy));

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
    d->tuneModeText = policy.modeText;

    try {
      char fp[512];
      // The backend hands back BOTH the identity string and the raw subgroup size in one call, so
      // "the sg inside the string" and "the sg the row path uses" cannot drift (G-C3: the shell has
      // no device handle, so the wave can only be INJECTED, and this call is the seam that already
      // existed).  `sg` stays RAW here: vk_tile_wave_for() is the single conversion and it is called
      // where the wave is used, never here.
      uint32_t sg = 0;
      const size_t fpLen = backendFingerprint(backend, fp, sizeof(fp), &sg);
      if (fpLen == 0) {
        logLine("WARNING: the backend could not describe the device; the tuner is disabled for device "
                + to_string(deviceIdx) + " (fail-open: no tuned rows)");
        return nullptr;
      }
      if (sg == 0) {
        // Fail-closed, in the same direction as the predicate's `wave == 0` clause: a device whose
        // wave we do not know cannot have its geometry judged, and assuming 32 would silently refuse
        // every legal wave64 row (or accept illegal wave32 ones).  Loud, and fail-open (no rows).
        logLine("WARNING: the backend reported subgroup size 0 for device " + to_string(deviceIdx)
                + "; the tuner is disabled for it (a geometry cannot be judged without the wave; "
                  "fail-open: no tuned rows)");
        return nullptr;
      }
      d->deviceSubgroupSize = sg;
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

      // Step 3-4's self-check, once per process, under the backend's probe bit 8.  It runs HERE
      // because it needs a real cache directory (to prove nothing was written into it) and the
      // resolved policy; it is deliberately before the load decides anything, so its line always
      // precedes the decision line in the log (order is what makes a two-line story readable).
      if (t2ProbeBit8()) {
        static std::atomic<bool> selfTested {false};
        bool expected = false;
        if (selfTested.compare_exchange_strong(expected, true)) {
          cacheWriteSelfTest(policy, d->cacheDir);
        }
      }

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
      else if (retune) {
        // Loudly, because "I set RETUNE and nothing happened" must never be the outcome, and
        // because the reader has to be able to tell this leg apart from a normal load.
        d->cacheState = "retune-ignored";
        d->rowSource = "none";
        d->note = string("KATAGO_VK_TUNE_RETUNE is set: ") + loadKind
                + " ignored entirely (rows and negative entry), deciding again";
        logLine("WARNING: " + string("KATAGO_VK_TUNE_RETUNE is set, so ") + loadKind + " ("
                + loadPath + ") is IGNORED: neither its rows nor its negative entry are used"
                " (delete the env to use it again)");
      }
      else {
        RowsFile file;
        string err;
        string text;
        if (!readWholeFile(loadPath, text, err) || !parseRowsFile(text, d->deviceSubgroupSize, file, err)) {
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
              // used.  But it must be LOUD: silently accepting a file that carries a hash for
              // different rows is exactly the "it looked fine" trap (lesson #46).
              logLine("WARNING: " + string(loadKind) + " declares ROWSET_HASH "
                      + file.declaredRowSetHash + " but the parsed rows hash to "
                      + rowSetHash(file.rows) + "; using the parsed rows"
                      " (IMPORT treats the declared hash as a hint)");
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
            if (d->negative) {
              // Carried onto the Decision, not just into the note: EXPORT has to re-emit the
              // marker (it writes what a cache file would contain), and the cache writer has to
              // check the reason against the closed set.  In the "both ROW and NEGATIVE" case
              // below the reason is deliberately dropped with the stale entry.
              d->negativeReason = file.negativeReason;
            }
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
        // WRITE-PATH: export
        // Declared for guardrail R16.  An export writes a file the OPERATOR named, never the
        // cache, so it is allowed even in read-only mode (vulkanTune=off) -- which is why this
        // is the one write path that does not have to sit behind a policy.off gate.  The cache
        // write path (step 3) carries its own tag and MUST be gated.
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
    // the context), so this side reports a failure, and -- B1 -- the IMPORT face: an
    // imported row set is USER DATA that decides which production pipelines get built, and
    // the one thing it must never do is change behaviour without saying so (lesson #46).
    const vk_tune_row_spec* rows = decision->rows.empty() ? nullptr : decision->rows.data();
    vk_tune_install_result installed {};
    string err;
    if (!backendInstallRows(backend, rows, decision->rows.size(), err, installed)) {
      logLine("WARNING: install failed for " + decision->deviceName + ": " + err
              + " (fail-open: this context uses the stock heuristic)");
      return;
    }
    // Conservative accumulation over this device's contexts: a row belongs to the durable
    // set only if EVERY context installed it, so keep the smallest `installed` and the
    // largest drop counts (see the field comment in the header).
    if (!decision->installSeen) {
      decision->install = installed;
      decision->installSeen = true;
    } else {
      decision->install.installed = std::min(decision->install.installed, installed.installed);
      decision->install.dropped_illegal = std::max(decision->install.dropped_illegal, installed.dropped_illegal);
      decision->install.dropped_unsupported = std::max(decision->install.dropped_unsupported, installed.dropped_unsupported);
      decision->install.dropped_build_failed = std::max(decision->install.dropped_build_failed, installed.dropped_build_failed);
      // The two per-target fields are inside no equality (they describe one row twice), so
      // the conservative direction for them is the maximum: a partial install or a failed
      // variant anywhere disqualifies the stronger claim.
      decision->install.partial = std::max(decision->install.partial, installed.partial);
      decision->install.variant_failed = std::max(decision->install.variant_failed, installed.variant_failed);
    }
    if (decision->imported && installed.declared > 0) {
      logLine("IMPORT installed " + to_string(installed.installed) + " of "
              + to_string(installed.declared) + " row(s) from " + decision->importedFrom
              + " (not key-checked: IMPORT is a reflow/development channel, not evidence that we measured it)");
    }
  }

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND
