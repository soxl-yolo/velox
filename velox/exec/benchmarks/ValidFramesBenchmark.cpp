/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Microbenchmark for computeValidFrames (velox/exec/Window.cpp).
//
// computeValidFrames runs once per output batch for every window function with
// a k-rows or k-range frame. It validates three predicates on each row's frame
// bounds (frameEnd >= 0, frameStart <= lastRow, frameStart <= frameEnd) and
// clamps valid bounds into [0, lastRow].
//
// This benchmark compares the scalar baseline (pre-56afedcfd) against the
// AVX2-SIMD path (56afedcfd) that processes 8 int32 frame-bound pairs per
// cycle. Three scenarios cover the common case (no violations, all rows
// already valid) and less common cases (frame bound violations, rows already
// marked invalid by a prior step).
//
// Run:
//   velox_valid_frames_benchmark --bm_min_iters=1000

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "velox/common/base/SimdUtil.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/TypeAliases.h"

using namespace facebook::velox;

namespace {

// ---------------------------------------------------------------------------
// Scalar baseline (identical to Window.cpp before 56afedcfd)
// ---------------------------------------------------------------------------
void computeValidFramesScalar(
    vector_size_t lastRow,
    vector_size_t numRows,
    vector_size_t* rawFrameStarts,
    vector_size_t* rawFrameEnds,
    SelectivityVector& validFrames) {
  for (auto i = 0; i < numRows; ++i) {
    if (!validFrames.isValid(i)) {
      continue;
    }
    const vector_size_t frameStart = rawFrameStarts[i];
    const vector_size_t frameEnd = rawFrameEnds[i];
    if (frameStart <= frameEnd && frameEnd >= 0 && frameStart <= lastRow) {
      rawFrameStarts[i] = std::max(frameStart, 0);
      rawFrameEnds[i] = std::min(frameEnd, lastRow);
    } else {
      validFrames.setValid(i, false);
    }
  }
  validFrames.updateBounds();
}

#if XSIMD_WITH_AVX2
// ---------------------------------------------------------------------------
// AVX2-SIMD implementation (56afedcfd)
// ---------------------------------------------------------------------------
void computeValidFramesSimd(
    vector_size_t lastRow,
    vector_size_t numRows,
    vector_size_t* rawFrameStarts,
    vector_size_t* rawFrameEnds,
    SelectivityVector& validFrames) {
  using batch_i32 = xsimd::batch<int32_t>;
  static_assert(batch_i32::size == 8, "");

  const uint8_t* validBitBytes =
      reinterpret_cast<const uint8_t*>(validFrames.allBits());
  const batch_i32 kZero(0);
  const batch_i32 kLastRow(lastRow);

  vector_size_t i = 0;
  for (; i + 8 <= numRows; i += 8) {
    const uint8_t existingBits = validBitBytes[i >> 3];
    if (existingBits == 0) {
      continue;
    }

    const batch_i32 starts = batch_i32::load_unaligned(rawFrameStarts + i);
    const batch_i32 ends = batch_i32::load_unaligned(rawFrameEnds + i);

    const auto valid =
        (ends >= kZero) & (starts <= kLastRow) & (starts <= ends);

    xsimd::select(valid, xsimd::max(starts, kZero), starts)
        .store_unaligned(rawFrameStarts + i);
    xsimd::select(valid, xsimd::min(ends, kLastRow), ends)
        .store_unaligned(rawFrameEnds + i);

    const uint8_t passedBits =
        static_cast<uint8_t>(simd::toBitMask<int32_t>(valid));
    uint8_t nowInvalid = existingBits & ~passedBits;
    while (nowInvalid) {
      const int j = __builtin_ctz(static_cast<uint32_t>(nowInvalid));
      validFrames.setValid(i + j, false);
      nowInvalid &= nowInvalid - 1;
    }
  }

  for (; i < numRows; ++i) {
    if (!validFrames.isValid(i)) {
      continue;
    }
    const vector_size_t frameStart = rawFrameStarts[i];
    const vector_size_t frameEnd = rawFrameEnds[i];
    if (frameStart <= frameEnd && frameEnd >= 0 && frameStart <= lastRow) {
      rawFrameStarts[i] = std::max(frameStart, 0);
      rawFrameEnds[i] = std::min(frameEnd, lastRow);
    } else {
      validFrames.setValid(i, false);
    }
  }

  validFrames.updateBounds();
}
#endif // XSIMD_WITH_AVX2

// ---------------------------------------------------------------------------
// Test-data fixture
// ---------------------------------------------------------------------------

// Encapsulates a snapshot of frame-bound arrays and validity bits that are
// copied into working buffers before each benchmark iteration (inside
// BENCHMARK_SUSPEND so the copy doesn't count toward timing).
struct TestData {
  vector_size_t lastRow;
  // Snapshot templates – never modified by the benchmark.
  std::vector<vector_size_t> startsTemplate;
  std::vector<vector_size_t> endsTemplate;
  SelectivityVector validTemplate;
  // Working copies – reset from templates before each timed iteration.
  std::vector<vector_size_t> starts;
  std::vector<vector_size_t> ends;
  SelectivityVector valid;

  // numRows: output batch size.
  // violationRate: fraction of rows whose frameEnd is set to -1 (fails P1).
  // preInvalidRate: fraction of rows pre-marked invalid in the SelectivityVector
  //   (simulates a prior step that already filtered some rows).
  TestData(
      vector_size_t numRows,
      double violationRate,
      double preInvalidRate = 0.0)
      : lastRow(numRows - 1),
        startsTemplate(numRows),
        endsTemplate(numRows),
        validTemplate(numRows, true),
        starts(numRows),
        ends(numRows),
        valid(numRows, true) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> rowDist(0, lastRow);
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    for (auto i = 0; i < numRows; ++i) {
      if (prob(rng) < violationRate) {
        // Invalid frame: frameEnd < 0 (violates predicate P1).
        startsTemplate[i] = rowDist(rng);
        endsTemplate[i] = -1;
      } else {
        // Valid frame that may need clamping (start slightly below 0 or end
        // slightly above lastRow, which is the common output of
        // updateKRowsFrameBounds for boundary rows).
        const auto mid = rowDist(rng);
        const auto half = std::min(mid, lastRow - mid);
        startsTemplate[i] = mid - half - 3; // may be negative → clamped to 0
        endsTemplate[i] = mid + half + 3;   // may exceed lastRow → clamped
      }
      if (prob(rng) < preInvalidRate) {
        validTemplate.setValid(i, false);
      }
    }
    validTemplate.updateBounds();
    reset();
  }

  void reset() {
    starts = startsTemplate;
    ends = endsTemplate;
    valid.setFromBits(validTemplate.allBits(), validTemplate.size());
  }

  vector_size_t numRows() const {
    return static_cast<vector_size_t>(starts.size());
  }
};

// ---------------------------------------------------------------------------
// Global fixtures initialised in main()
// ---------------------------------------------------------------------------

// Scenario A – all rows valid, no frame violations.  This is the common case
// for most window queries; it tests pure predicate + clamp throughput.
std::unique_ptr<TestData> gNoViolations256;
std::unique_ptr<TestData> gNoViolations1024;
std::unique_ptr<TestData> gNoViolations4096;

// Scenario B – half the rows have an invalid frame bound (frameEnd = -1).
// Tests the path where setValid(false) is called for violating rows.
std::unique_ptr<TestData> gHalfViolations1024;

// Scenario C – all rows are pre-marked invalid in the SelectivityVector.
// Tests the fast-path that skips SIMD work for fully-invalid 8-row batches.
std::unique_ptr<TestData> gAllPreInvalid1024;

// ---------------------------------------------------------------------------
// Macro helpers to define a scalar + SIMD pair
// ---------------------------------------------------------------------------

#define BENCH_SCALAR(name, fixture)                           \
  BENCHMARK(scalar_##name) {                                  \
    BENCHMARK_SUSPEND { (fixture)->reset(); }                 \
    computeValidFramesScalar(                                 \
        (fixture)->lastRow,                                   \
        (fixture)->numRows(),                                 \
        (fixture)->starts.data(),                             \
        (fixture)->ends.data(),                               \
        (fixture)->valid);                                    \
  }

#if XSIMD_WITH_AVX2
#define BENCH_SIMD(name, fixture)                             \
  BENCHMARK_RELATIVE(simd_##name) {                          \
    BENCHMARK_SUSPEND { (fixture)->reset(); }                 \
    computeValidFramesSimd(                                   \
        (fixture)->lastRow,                                   \
        (fixture)->numRows(),                                 \
        (fixture)->starts.data(),                             \
        (fixture)->ends.data(),                               \
        (fixture)->valid);                                    \
  }
#else
#define BENCH_SIMD(name, fixture) // no AVX2
#endif

} // namespace

// ============================================================================
// Scenario A: all rows valid, no violations (common case)
// ============================================================================

BENCH_SCALAR(noViolations_256, gNoViolations256)
BENCH_SIMD(noViolations_256, gNoViolations256)
BENCHMARK_DRAW_LINE();

BENCH_SCALAR(noViolations_1024, gNoViolations1024)
BENCH_SIMD(noViolations_1024, gNoViolations1024)
BENCHMARK_DRAW_LINE();

BENCH_SCALAR(noViolations_4096, gNoViolations4096)
BENCH_SIMD(noViolations_4096, gNoViolations4096)
BENCHMARK_DRAW_LINE();

// ============================================================================
// Scenario B: half the rows have invalid frame bounds
// ============================================================================

BENCH_SCALAR(halfViolations_1024, gHalfViolations1024)
BENCH_SIMD(halfViolations_1024, gHalfViolations1024)
BENCHMARK_DRAW_LINE();

// ============================================================================
// Scenario C: all rows already marked invalid (fast-path skip)
// ============================================================================

BENCH_SCALAR(allPreInvalid_1024, gAllPreInvalid1024)
BENCH_SIMD(allPreInvalid_1024, gAllPreInvalid1024)

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  // Scenario A
  gNoViolations256 = std::make_unique<TestData>(256, /*violationRate=*/0.0);
  gNoViolations1024 = std::make_unique<TestData>(1024, /*violationRate=*/0.0);
  gNoViolations4096 = std::make_unique<TestData>(4096, /*violationRate=*/0.0);

  // Scenario B
  gHalfViolations1024 =
      std::make_unique<TestData>(1024, /*violationRate=*/0.5);

  // Scenario C: preInvalidRate=1.0 → every row pre-marked invalid.
  gAllPreInvalid1024 =
      std::make_unique<TestData>(1024, /*violationRate=*/0.0, /*preInvalidRate=*/1.0);

  folly::runBenchmarks();
  return 0;
}
