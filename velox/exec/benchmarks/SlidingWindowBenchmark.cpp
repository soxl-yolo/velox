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

// Microbenchmark for the two-stack sliding window aggregation (7db3cab52).
//
// Before 7db3cab52, ROWS BETWEEN K PRECEDING AND CURRENT ROW recomputed the
// aggregate from scratch for every output row: O(K) per row, O(N·K) total.
// 7db3cab52 introduces a two-stack (deque) algorithm that amortises cost to
// O(1) per row, O(N) total.
//
// Key experiment: fix partition size N and vary K. With the two-stack the
// wall-clock time is flat across K values; without it the time would grow
// proportionally to K. The benchmark also compares against UNBOUNDED PRECEDING
// (the pre-existing O(N) incremental path) as a throughput ceiling.
//
// Benchmark groups
// ────────────────
// Group A – vary K, fixed N=100 K rows, sum(v):
//   slidingWindowSum_k10    ROWS BETWEEN    10 PRECEDING AND CURRENT ROW
//   slidingWindowSum_k100   ROWS BETWEEN   100 PRECEDING AND CURRENT ROW
//   slidingWindowSum_k1000  ROWS BETWEEN  1000 PRECEDING AND CURRENT ROW
//   slidingWindowSum_k10000 ROWS BETWEEN 10000 PRECEDING AND CURRENT ROW
//   unboundedSum            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
//
// Group B – avg(v) at the same K values (avg has a complex ROW intermediate
//   type, exercising the intermediate-type resolution added in 7db3cab52).
//
// Group C – vary N, fixed K=100: shows O(N) scaling for both sum and avg.
//
// Run:
//   velox_sliding_window_benchmark --bm_min_iters=5

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

constexpr int32_t kInputBatchRows = 10'000;
constexpr int32_t kOutputBatchRows = 1'000;

class SlidingWindowBenchmark : public VectorTestBase {
 public:
  void addBenchmarks() {
    // Group A: vary K, N=100 K, sum(v).
    addCase("slidingWindowSum_k10", 100'000, 10, "sum(v)");
    addCase("slidingWindowSum_k100", 100'000, 100, "sum(v)");
    addCase("slidingWindowSum_k1000", 100'000, 1'000, "sum(v)");
    addCase("slidingWindowSum_k10000", 100'000, 10'000, "sum(v)");
    addCase("unboundedSum", 100'000, -1 /*unbounded*/, "sum(v)");

    // Group B: vary K, N=100 K, avg(v) — tests ROW intermediate type.
    addCase("slidingWindowAvg_k10", 100'000, 10, "avg(v)");
    addCase("slidingWindowAvg_k100", 100'000, 100, "avg(v)");
    addCase("slidingWindowAvg_k1000", 100'000, 1'000, "avg(v)");
    addCase("slidingWindowAvg_k10000", 100'000, 10'000, "avg(v)");
    addCase("unboundedAvg", 100'000, -1, "avg(v)");

    // Group C: vary N, K=100, sum(v).
    addCase("slidingWindowSum_N10K_k100", 10'000, 100, "sum(v)");
    addCase("slidingWindowSum_N100K_k100", 100'000, 100, "sum(v)");
    addCase("slidingWindowSum_N500K_k100", 500'000, 100, "sum(v)");
  }

 private:
  struct TestCase {
    std::string name;
    std::vector<RowVectorPtr> data;
    core::PlanNodePtr plan;
    int64_t numRows;
  };

  // Builds a single-partition dataset of numRows rows: one value column v.
  std::vector<RowVectorPtr> makeData(int64_t numRows) {
    const auto rowType = ROW({"s", "v"}, {INTEGER(), BIGINT()});
    std::vector<RowVectorPtr> result;
    result.reserve((numRows + kInputBatchRows - 1) / kInputBatchRows);
    int64_t remaining = numRows;
    int64_t offset = 0;
    while (remaining > 0) {
      const int32_t batchRows =
          static_cast<int32_t>(std::min<int64_t>(remaining, kInputBatchRows));
      result.push_back(makeRowVector(
          rowType->names(),
          {
              makeFlatVector<int32_t>(
                  batchRows, [offset](auto row) { return offset + row; }),
              makeFlatVector<int64_t>(
                  batchRows, [](auto row) { return row % 97; }),
          }));
      offset += batchRows;
      remaining -= batchRows;
    }
    return result;
  }

  // Builds the window expression. k == -1 means UNBOUNDED PRECEDING.
  static std::string windowExpr(const std::string& agg, int32_t k) {
    const std::string frame = (k < 0)
        ? "rows between unbounded preceding and current row"
        : "rows between " + std::to_string(k) + " preceding and current row";
    return agg + " over (order by s " + frame + ")";
  }

  void addCase(
      const std::string& name,
      int64_t numRows,
      int32_t k,
      const std::string& agg) {
    auto tc = std::make_unique<TestCase>();
    tc->name = name;
    tc->numRows = numRows;
    tc->data = makeData(numRows);
    // Use streamingWindow: data is already sorted by s (single partition).
    tc->plan = exec::test::PlanBuilder()
                   .values(tc->data)
                   .streamingWindow({windowExpr(agg, k)})
                   .planNode();

    const auto* raw = tc.get();
    folly::addBenchmark(
        __FILE__,
        name,
        [this, raw](folly::UserCounters& counters, unsigned iterations) {
          CpuWallTiming timing;
          uint64_t totalRows = 0;
          for (unsigned i = 0; i < iterations; ++i) {
            totalRows += runOnce(*raw, timing);
          }
          BENCHMARK_SUSPEND {
            counters["rows"] = folly::UserMetric(
                static_cast<int64_t>(raw->numRows),
                folly::UserMetric::Type::METRIC);
            counters["windowCpuSec"] = folly::UserMetric(
                static_cast<double>(timing.cpuNanos) / iterations / 1e9,
                folly::UserMetric::Type::TIME);
            counters["windowWallSec"] = folly::UserMetric(
                static_cast<double>(timing.wallNanos) / iterations / 1e9,
                folly::UserMetric::Type::TIME);
            folly::doNotOptimizeAway(totalRows);
          }
          return iterations;
        });

    cases_.push_back(std::move(tc));
  }

  uint64_t runOnce(const TestCase& tc, CpuWallTiming& timing) const {
    std::unique_ptr<TaskCursor> cursor;
    BENCHMARK_SUSPEND {
      CursorParameters params;
      params.planNode = tc.plan;
      params.serialExecution = true;
      params.queryConfigs = {
          {core::QueryConfig::kPreferredOutputBatchRows,
           std::to_string(kOutputBatchRows)}};
      cursor = TaskCursor::create(params);
    }

    uint64_t numRows = 0;
    while (cursor->moveNext()) {
      numRows += cursor->current()->size();
    }

    BENCHMARK_SUSPEND {
      VELOX_CHECK_EQ(numRows, tc.numRows);
      const auto stats = cursor->task()->taskStats();
      for (const auto& pipeline : stats.pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          if (op.operatorType == OperatorType::kWindow) {
            timing.add(op.addInputTiming);
            timing.add(op.getOutputTiming);
          }
        }
      }
    }
    return numRows;
  }

  std::vector<std::unique_ptr<TestCase>> cases_;
};

std::unique_ptr<SlidingWindowBenchmark> benchmark;

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  aggregate::prestosql::registerAllAggregateFunctions();
  window::prestosql::registerAllWindowFunctions();
  benchmark = std::make_unique<SlidingWindowBenchmark>();
  benchmark->addBenchmarks();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
