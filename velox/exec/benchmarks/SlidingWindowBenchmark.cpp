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
// The benchmark has three groups.
//
// Group A – before vs after, fixed N=100 K, sum(v):
//   For each K in {10, 100, 1000, 10000}, a BENCHMARK_RELATIVE pair is shown:
//     naive_sum_kK   – window_sliding_agg_disabled=true  (O(N·K) recompute)
//     twoStack_sum_kK – window_sliding_agg_disabled=false (O(N) two-stack)
//   With the two-stack the "relative" column should be ~K× for large K.
//
// Group B – same for avg(v), which uses a ROW(sum,count) intermediate type.
//
// Group C – vary N at K=100 to confirm O(N) linear scaling for the two-stack.
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
    // Group A: naive vs two-stack, N=100 K, sum(v).
    for (int32_t k : {10, 100, 1'000, 10'000}) {
      const std::string ks = std::to_string(k);
      addCase("naive_sum_k" + ks, 100'000, k, "sum(v)", /*disableSliding=*/true);
      addRelCase("twoStack_sum_k" + ks, 100'000, k, "sum(v)", /*disableSliding=*/false);
    }

    // Group B: naive vs two-stack, N=100 K, avg(v).
    for (int32_t k : {10, 100, 1'000, 10'000}) {
      const std::string ks = std::to_string(k);
      addCase("naive_avg_k" + ks, 100'000, k, "avg(v)", /*disableSliding=*/true);
      addRelCase("twoStack_avg_k" + ks, 100'000, k, "avg(v)", /*disableSliding=*/false);
    }

    // Group C: two-stack scaling in N at K=100.
    addCase("twoStack_sum_N10K_k100", 10'000, 100, "sum(v)", false);
    addCase("twoStack_sum_N100K_k100", 100'000, 100, "sum(v)", false);
    addCase("twoStack_sum_N500K_k100", 500'000, 100, "sum(v)", false);
  }

 private:
  struct TestCase {
    std::string name;
    std::vector<RowVectorPtr> data;
    core::PlanNodePtr plan;
    bool disableSlidingAgg;
    int64_t numRows;
  };

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

  // k == -1 means UNBOUNDED PRECEDING.
  static std::string windowExpr(const std::string& agg, int32_t k) {
    const std::string frame = (k < 0)
        ? "rows between unbounded preceding and current row"
        : "rows between " + std::to_string(k) + " preceding and current row";
    return agg + " over (order by s " + frame + ")";
  }

  void registerCase(
      const std::string& name,
      int64_t numRows,
      int32_t k,
      const std::string& agg,
      bool disableSlidingAgg,
      bool relative) {
    auto tc = std::make_unique<TestCase>();
    tc->name = name;
    tc->numRows = numRows;
    tc->disableSlidingAgg = disableSlidingAgg;
    tc->data = makeData(numRows);
    tc->plan = exec::test::PlanBuilder()
                   .values(tc->data)
                   .streamingWindow({windowExpr(agg, k)})
                   .planNode();

    const auto* raw = tc.get();
    auto fn = [this, raw](
                  folly::UserCounters& counters, unsigned iterations) {
      CpuWallTiming timing;
      uint64_t totalRows = 0;
      for (unsigned i = 0; i < iterations; ++i) {
        totalRows += runOnce(*raw, timing);
      }
      BENCHMARK_SUSPEND {
        counters["rows"] = folly::UserMetric(
            static_cast<int64_t>(raw->numRows),
            folly::UserMetric::Type::METRIC);
        counters["wallSec"] = folly::UserMetric(
            static_cast<double>(timing.wallNanos) / iterations / 1e9,
            folly::UserMetric::Type::TIME);
        folly::doNotOptimizeAway(totalRows);
      }
      return iterations;
    };

    if (relative) {
      folly::addBenchmark(__FILE__, "%" + name, fn);
    } else {
      folly::addBenchmark(__FILE__, name, fn);
    }

    cases_.push_back(std::move(tc));
  }

  void addCase(
      const std::string& name,
      int64_t numRows,
      int32_t k,
      const std::string& agg,
      bool disableSlidingAgg) {
    registerCase(name, numRows, k, agg, disableSlidingAgg, /*relative=*/false);
  }

  void addRelCase(
      const std::string& name,
      int64_t numRows,
      int32_t k,
      const std::string& agg,
      bool disableSlidingAgg) {
    registerCase(name, numRows, k, agg, disableSlidingAgg, /*relative=*/true);
  }

  uint64_t runOnce(const TestCase& tc, CpuWallTiming& timing) const {
    std::unique_ptr<TaskCursor> cursor;
    BENCHMARK_SUSPEND {
      CursorParameters params;
      params.planNode = tc.plan;
      params.serialExecution = true;
      params.queryConfigs = {
          {core::QueryConfig::kPreferredOutputBatchRows,
           std::to_string(kOutputBatchRows)},
          {core::QueryConfig::kWindowSlidingAggDisabled,
           tc.disableSlidingAgg ? "true" : "false"},
      };
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
