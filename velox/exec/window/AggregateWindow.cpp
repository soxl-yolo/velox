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

#include "velox/exec/window/AggregateWindow.h"
#include "velox/common/base/Exceptions.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec::window {

namespace {

/// Maintains aggregate state for a sliding window frame using the two-stack
/// (double-ended queue) algorithm. Achieves O(1) amortized cost per row
/// advancement instead of O(N) per row for naive recomputation.
///
/// Requires both frameStart and frameEnd to be monotonically non-decreasing
/// across output rows. Supports all associative aggregates with no changes to
/// the Aggregate interface.
///
/// Algorithm invariants:
///   back_[i].prefix = agg(back_[0].raw, ..., back_[i].raw) in window order.
///   front_[0] = newest element in front; front_.back() = oldest (next to pop).
///   front_[i].suffix = agg(front_[i], ..., front_[0]) in window order,
///     where front_[i] is older than front_[i-1].
///   Window aggregate = combine(front_.back().suffix, back_.back().prefix).
///
/// Uses rawAgg (kSingle step) for raw input and final extraction, and
/// mergeAgg (kPartial step) for extractAccumulators. Both share the same
/// accumulator row layout (guaranteed by Velox's Aggregate contract).
class SlidingWindowAgg {
 public:
  SlidingWindowAgg(
      Aggregate* rawAgg,
      Aggregate* mergeAgg,
      int32_t rowSize,
      memory::MemoryPool* pool,
      const TypePtr& intermediateType)
      : rawAgg_(rawAgg),
        mergeAgg_(mergeAgg),
        rowSize_(rowSize),
        pool_(pool) {
    mergeVec_ = BaseVector::create(intermediateType, 1, pool_);
  }

  ~SlidingWindowAgg() {
    destroyLiveSlots();
  }

  /// Resets the two-stack state for a new partition.
  void reset() {
    destroyLiveSlots();
    front_.clear();
    back_.clear();
  }

  bool empty() const {
    return front_.empty() && back_.empty();
  }

  /// Adds row 0 of argVectors to the right end of the window.
  /// The caller is responsible for loading exactly one row at position 0.
  void pushBack(const std::vector<VectorPtr>& argVectors) {
    static const SelectivityVector kRow0(1);

    char* raw = obtainSlot();
    rawAgg_->addSingleGroupRawInput(raw, kRow0, argVectors, false);

    char* prefix = obtainSlot();
    if (back_.empty()) {
      copyAccum(prefix, raw);
    } else {
      copyAccum(prefix, back_.back().prefix);
      mergeAccum(prefix, raw);
    }
    back_.push_back({raw, prefix});
  }

  /// Removes the oldest (leftmost) row from the window.
  void popFront() {
    VELOX_CHECK(!empty(), "Cannot pop from empty sliding window");
    if (front_.empty()) {
      rebuildFront();
    }
    returnSlot(front_.back());
    front_.pop_back();
  }

  /// Extracts the current window aggregate into result[resultOffset].
  void extractResult(
      VectorPtr& aggResultVec,
      const VectorPtr& result,
      vector_size_t resultOffset) {
    char* source = nullptr;
    char* merged = nullptr;

    if (front_.empty()) {
      source = back_.back().prefix;
    } else if (back_.empty()) {
      source = front_.back();
    } else {
      merged = obtainSlot();
      copyAccum(merged, front_.back());
      mergeAccum(merged, back_.back().prefix);
      source = merged;
    }

    BaseVector::prepareForReuse(aggResultVec, 1);
    rawAgg_->extractValues(&source, 1, &aggResultVec);
    result->copy(aggResultVec.get(), resultOffset, 0, 1);

    if (merged != nullptr) {
      returnSlot(merged);
    }
  }

 private:
  struct BackEntry {
    /// Accumulator for just this row's raw input.
    char* raw;
    /// Prefix cumulative: agg(back_[0..this]) in window order.
    char* prefix;
  };

  /// kSingle step aggregate: used for raw input and final extraction.
  Aggregate* rawAgg_;
  /// kPartial step aggregate: used for extractAccumulators (merging).
  /// Both share the same accumulator row layout.
  Aggregate* mergeAgg_;
  int32_t rowSize_;
  memory::MemoryPool* pool_;

  /// front_[0] = newest in front (last to leave).
  /// front_.back() = oldest in front (next to pop).
  /// front_[i].suffix = agg(front_[i], ..., front_[0]) in window order.
  std::vector<char*> front_;

  /// back_[0] = oldest in back. back_.back() = newest.
  std::vector<BackEntry> back_;

  /// All ever-allocated accumulator buffers; keeps memory alive.
  std::vector<BufferPtr> allBuffers_;
  /// Destroyed slots available for reuse.
  std::vector<char*> freeList_;

  /// Temporary vector for accumulator merge operations.
  VectorPtr mergeVec_;

  /// Rebuilds the front stack from all back entries when front is empty.
  /// After rebuild, front_[0] is the newest element (from back_.back()),
  /// and front_.back() is the oldest (from back_[0]).
  void rebuildFront() {
    VELOX_CHECK(!back_.empty());
    size_t n = back_.size();
    std::vector<char*> newFront(n);

    // front_[0].suffix = agg(just back_[n-1].raw).
    // front_[i].suffix = combine(back_[n-1-i].raw, front_[i-1].suffix).
    newFront[0] = obtainSlot();
    copyAccum(newFront[0], back_[n - 1].raw);

    for (size_t i = 1; i < n; ++i) {
      newFront[i] = obtainSlot();
      copyAccum(newFront[i], back_[n - 1 - i].raw);
      mergeAccum(newFront[i], newFront[i - 1]);
    }

    for (auto& entry : back_) {
      returnSlot(entry.raw);
      returnSlot(entry.prefix);
    }
    back_.clear();
    front_ = std::move(newFront);
  }

  /// Allocates and initializes a fresh accumulator slot, reusing freed slots.
  char* obtainSlot() {
    char* row = nullptr;
    if (!freeList_.empty()) {
      row = freeList_.back();
      freeList_.pop_back();
    } else {
      allBuffers_.push_back(AlignedBuffer::allocate<char>(rowSize_, pool_));
      row = allBuffers_.back()->asMutable<char>();
    }
    static const vector_size_t kZero{0};
    rawAgg_->initializeNewGroups(&row, folly::Range(&kZero, 1));
    return row;
  }

  /// Destroys out-of-line accumulator state and returns the slot to the free
  /// list.
  void returnSlot(char* row) {
    rawAgg_->destroy(folly::Range(&row, 1));
    freeList_.push_back(row);
  }

  /// Copies src's entire accumulator state into dst (dst must be freshly
  /// initialized).
  void copyAccum(char* dst, char* src) {
    BaseVector::prepareForReuse(mergeVec_, 1);
    mergeAgg_->extractAccumulators(&src, 1, &mergeVec_);
    static const SelectivityVector kSingleRow(1);
    rawAgg_->addSingleGroupIntermediateResults(dst, kSingleRow, {mergeVec_}, false);
  }

  /// Merges src's accumulator state into dst in window order (dst = agg(dst,
  /// src)).
  void mergeAccum(char* dst, char* src) {
    BaseVector::prepareForReuse(mergeVec_, 1);
    mergeAgg_->extractAccumulators(&src, 1, &mergeVec_);
    static const SelectivityVector kSingleRow(1);
    rawAgg_->addSingleGroupIntermediateResults(dst, kSingleRow, {mergeVec_}, false);
  }

  void destroyLiveSlots() {
    for (auto* slot : front_) {
      returnSlot(slot);
    }
    for (auto& entry : back_) {
      returnSlot(entry.raw);
      returnSlot(entry.prefix);
    }
  }
};

// A generic way to compute any aggregation used as a window function.
// Creates an Aggregate function object for the window function invocation.
// At each row, computes the aggregation across all rows from the frameStart
// to frameEnd boundaries at that row using singleGroup.
class AggregateWindowFunction : public exec::WindowFunction {
 public:
  AggregateWindowFunction(
      const std::string& name,
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      bool ignoreNulls,
      velox::memory::MemoryPool* pool,
      HashStringAllocator* stringAllocator,
      const core::QueryConfig& config)
      : WindowFunction(resultType, pool, stringAllocator) {
    VELOX_USER_CHECK(
        !ignoreNulls, "Aggregate window functions do not support IGNORE NULLS");
    argTypes_.reserve(args.size());
    argIndices_.reserve(args.size());
    argVectors_.reserve(args.size());
    for (const auto& arg : args) {
      argTypes_.push_back(arg.type);
      if (arg.constantValue) {
        argIndices_.push_back(kConstantChannel);
        argVectors_.push_back(arg.constantValue);
      } else {
        VELOX_CHECK(arg.index.has_value());
        argIndices_.push_back(arg.index.value());
        argVectors_.push_back(BaseVector::create(arg.type, 0, pool_));
      }
    }
    // Create an Aggregate function object to do result computation. Window
    // function usage only requires single group aggregation for calculating
    // the function value for each row.
    aggregate_ = exec::Aggregate::create(
        name,
        core::AggregationNode::Step::kSingle,
        argTypes_,
        resultType,
        config);
    aggregate_->setAllocator(stringAllocator_);

    // A kPartial aggregate with the same accumulator layout, used in the
    // two-stack sliding window path to extract intermediate accumulator state
    // for merging. The accumulator format is identical for kPartial and kSingle
    // per the Aggregate contract.
    //
    // For kPartial, the resultType parameter must be the INTERMEDIATE type
    // (e.g. ROW(sum, count) for avg), not the final result type. Passing the
    // wrong type causes extractAccumulators to produce incorrectly typed vectors.
    intermediateType_ = resolveIntermediateType(name, argTypes_);
    partialAggregate_ = exec::Aggregate::create(
        name,
        core::AggregationNode::Step::kPartial,
        argTypes_,
        intermediateType_,
        config);
    partialAggregate_->setAllocator(stringAllocator_);

    // Aggregate initialization.
    // Row layout is:
    //  - null flags - one bit per aggregate.
    //  - uint32_t row size,
    //  - fixed-width accumulators - one per aggregate
    //
    // Here we always make space for a row size since we only have one
    // row and no RowContainer. We also have a single aggregate here, so there
    // is only one null bit and one initialized bit.
    static const int32_t kAccumulatorFlagsOffset = 0;
    static const int32_t kRowSizeOffset = bits::nbytes(1);
    singleGroupRowSize_ = kRowSizeOffset + sizeof(int32_t);
    // Accumulator offset must be aligned by their alignment size.
    singleGroupRowSize_ = bits::roundUp(
        singleGroupRowSize_, aggregate_->accumulatorAlignmentSize());
    aggregate_->setOffsets(
        singleGroupRowSize_,
        exec::RowContainer::nullByte(kAccumulatorFlagsOffset),
        exec::RowContainer::nullMask(kAccumulatorFlagsOffset),
        exec::RowContainer::initializedByte(kAccumulatorFlagsOffset),
        exec::RowContainer::initializedMask(kAccumulatorFlagsOffset),
        /* needed for out of line allocations */ kRowSizeOffset);
    singleGroupRowSize_ += aggregate_->accumulatorFixedWidthSize();

    // Set the same offsets on partialAggregate_ so both share the row layout.
    partialAggregate_->setOffsets(
        singleGroupRowSize_ - aggregate_->accumulatorFixedWidthSize(),
        exec::RowContainer::nullByte(kAccumulatorFlagsOffset),
        exec::RowContainer::nullMask(kAccumulatorFlagsOffset),
        exec::RowContainer::initializedByte(kAccumulatorFlagsOffset),
        exec::RowContainer::initializedMask(kAccumulatorFlagsOffset),
        kRowSizeOffset);

    // Construct the single row in the MemoryPool.
    singleGroupRowBufferPtr_ =
        AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
    rawSingleGroupRow_ = singleGroupRowBufferPtr_->asMutable<char>();

    // Constructing a vector of a single result value used for copying from
    // the aggregate to the final result.
    aggregateResultVector_ = BaseVector::create(resultType, 1, pool_);

    slidingAggDisabled_ = config.windowSlidingAggDisabled();

    // Use a higher frame-width threshold for aggregates whose intermediate
    // type differs from the result type (e.g. avg: ROW(sum,count) vs DOUBLE).
    // Those aggregates pay a serialization cost on every push into the back
    // stack, so the two-stack breakeven is ~K=1000 instead of ~K=400.
    if (!intermediateType_->equivalent(*resultType)) {
      slidingWindowMinFrameWidth_ = 1000;
    }

    computeDefaultAggregateValue(resultType);
  }

  ~AggregateWindowFunction() {
    // Needed to delete any out-of-line storage for the accumulator in the
    // group row.
    if (aggregateInitialized_) {
      std::vector<char*> singleGroupRowVector = {rawSingleGroupRow_};
      aggregate_->destroy(folly::Range(singleGroupRowVector.data(), 1));
    }
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;
    previousFrameMetadata_.reset();
    partitionUseTwoStack_.reset();
    if (slidingWindowAgg_) {
      slidingWindowAgg_->reset();
    }
    rightPtr_ = 0;
    leftPtr_ = 0;
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds,
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) override {
    if (handleAllEmptyFrames(validRows, resultOffset, result)) {
      return;
    }

    auto rawFrameStarts = frameStarts->as<vector_size_t>();
    auto rawFrameEnds = frameEnds->as<vector_size_t>();

    FrameMetadata frameMetadata =
        analyzeFrameValues(validRows, rawFrameStarts, rawFrameEnds);

    if (frameMetadata.incrementalAggregation) {
      vector_size_t startRow;
      if (frameMetadata.usePreviousAggregate) {
        // If incremental aggregation can be resumed from the previous block,
        // then the argument vectors also can be populated from the previous
        // frameEnd to the current frameEnd. Only the new values are
        // required for computing aggregates.
        startRow = previousFrameMetadata_->lastRow + 1;
      } else {
        startRow = frameMetadata.firstRow;

        // This is the start of a new incremental aggregation. So the
        // aggregate_ function object should be initialized.
        auto singleGroup = std::vector<vector_size_t>{0};
        aggregate_->destroy(folly::Range<char**>(&rawSingleGroupRow_, 1));
        aggregate_->initializeNewGroups(&rawSingleGroupRow_, singleGroup);
        aggregateInitialized_ = true;
      }

      fillArgVectors(startRow, frameMetadata.lastRow);
      incrementalAggregation(
          validRows,
          startRow,
          frameMetadata.lastRow,
          rawFrameEnds,
          resultOffset,
          result);
    } else if (frameMetadata.slidingWindow && !slidingAggDisabled_) {
      // Decide once per partition whether the frame is wide enough to amortise
      // the two-stack overhead. Measure the average frame width on the first
      // sliding-window batch and cache the result for the rest of the partition.
      if (!partitionUseTwoStack_.has_value()) {
        int64_t totalWidth = 0;
        int32_t cnt = 0;
        validRows.applyToSelected([&](auto i) {
          totalWidth += rawFrameEnds[i] - rawFrameStarts[i] + 1;
          ++cnt;
        });
        const double avgWidth = cnt > 0 ? static_cast<double>(totalWidth) / cnt : 0;
        partitionUseTwoStack_ = (avgWidth >= slidingWindowMinFrameWidth_);
      }
      if (*partitionUseTwoStack_) {
        slidingWindowAggregation(
            validRows,
            frameMetadata,
            rawFrameStarts,
            rawFrameEnds,
            resultOffset,
            result);
      } else {
        fillArgVectors(frameMetadata.firstRow, frameMetadata.lastRow);
        simpleAggregation(
            validRows,
            frameMetadata.firstRow,
            frameMetadata.lastRow,
            rawFrameStarts,
            rawFrameEnds,
            resultOffset,
            result);
      }
    } else {
      fillArgVectors(frameMetadata.firstRow, frameMetadata.lastRow);
      simpleAggregation(
          validRows,
          frameMetadata.firstRow,
          frameMetadata.lastRow,
          rawFrameStarts,
          rawFrameEnds,
          resultOffset,
          result);
    }
    previousFrameMetadata_ = frameMetadata;
  }

 private:
  struct FrameMetadata {
    /// Min frame start row required for aggregation.
    vector_size_t firstRow;

    /// Max frame end required for the aggregation.
    vector_size_t lastRow;

    /// If all the rows in the block have the same start row, and the
    /// end frame rows are non-decreasing, then the aggregation can be done
    /// incrementally. With incremental aggregation new frame rows are
    /// accumulated over the previous result to obtain the new result.
    bool incrementalAggregation;

    /// Resume incremental aggregation from the prior block.
    bool usePreviousAggregate;

    /// Both frameStart and frameEnd are non-decreasing across rows, enabling
    /// the two-stack sliding window algorithm.
    bool slidingWindow;

    /// Continue the two-stack state from the prior block.
    bool usePreviousSlidingWindow;

    /// Last frame start in this batch, for cross-batch monotonicity checks.
    vector_size_t lastFrameStart;

    /// Last frame end in this batch, for cross-batch monotonicity checks.
    vector_size_t lastFrameEnd;
  };

  bool handleAllEmptyFrames(
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    if (!validRows.hasSelections()) {
      setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
      return true;
    }
    return false;
  }

  // Computes the least frameStart row and the max frameEnds row
  // indices for the valid frames of this output block. These indices are used
  // as bounds when reading input parameter vectors for aggregation.
  // This method expects to have at least 1 valid frame in the block.
  // Blocks with all empty frames are handled before this point.
  FrameMetadata analyzeFrameValues(
      const SelectivityVector& validRows,
      const vector_size_t* rawFrameStarts,
      const vector_size_t* rawFrameEnds) {
    VELOX_DCHECK(validRows.hasSelections());

    auto firstValidRow = validRows.begin();
    vector_size_t firstRow = rawFrameStarts[firstValidRow];
    vector_size_t fixedFrameStartRow = firstRow;
    vector_size_t lastRow = rawFrameEnds[firstValidRow];
    vector_size_t prevFrameStarts = firstRow;
    vector_size_t prevFrameEnds = lastRow;

    bool incrementalAggregation = true;
    bool startNonDecreasing = true;
    bool endNonDecreasing = true;

    validRows.applyToSelected([&](auto i) {
      firstRow = std::min(firstRow, rawFrameStarts[i]);
      lastRow = std::max(lastRow, rawFrameEnds[i]);

      // Incremental aggregation: fixed start and non-decreasing end.
      incrementalAggregation &= (rawFrameStarts[i] == fixedFrameStartRow);
      incrementalAggregation &= rawFrameEnds[i] >= prevFrameEnds;

      // Sliding window: both start and end are non-decreasing.
      startNonDecreasing &= (rawFrameStarts[i] >= prevFrameStarts);
      endNonDecreasing &= (rawFrameEnds[i] >= prevFrameEnds);

      prevFrameStarts = rawFrameStarts[i];
      prevFrameEnds = rawFrameEnds[i];
    });

    // Sliding window applies when both bounds are non-decreasing, but the
    // start is not fixed (otherwise the incremental path is preferred).
    // Restrict the two-stack path to aggregates with fixed-size accumulators.
    // For variable-size aggregates (e.g. array_agg, map_agg, histogram) the
    // prefix accumulator at back stack position i holds O(i) data, so W
    // prefixes together consume O(W²) space — worse than the O(W) state kept
    // by simpleAggregation. accumulatorUsesExternalMemory() is insufficient
    // because it only covers memory outside Velox's allocator; isFixedSize()
    // is the correct predicate for "accumulator size does not grow with input".
    bool slidingWindow = startNonDecreasing && endNonDecreasing &&
        !incrementalAggregation && aggregate_->isFixedSize();

    bool usePreviousAggregate = false;
    if (previousFrameMetadata_.has_value()) {
      auto& prev = previousFrameMetadata_.value();
      if (incrementalAggregation && prev.incrementalAggregation &&
          prev.firstRow == firstRow &&
          prev.lastRow <= rawFrameEnds[firstValidRow]) {
        usePreviousAggregate = true;
      }
    }

    bool usePreviousSlidingWindow = false;
    if (slidingWindow && previousFrameMetadata_.has_value()) {
      auto& prev = previousFrameMetadata_.value();
      // Continue the two-stack state if the previous batch also used it and
      // the frame boundaries are monotone at the batch boundary.
      if (prev.slidingWindow &&
          rawFrameStarts[firstValidRow] >= prev.lastFrameStart &&
          rawFrameEnds[firstValidRow] >= prev.lastFrameEnd) {
        usePreviousSlidingWindow = true;
      }
    }

    return {
        firstRow,
        lastRow,
        incrementalAggregation,
        usePreviousAggregate,
        slidingWindow,
        usePreviousSlidingWindow,
        prevFrameStarts,
        prevFrameEnds,
    };
  }

  void fillArgVectors(vector_size_t firstRow, vector_size_t lastRow) {
    vector_size_t numFrameRows = lastRow + 1 - firstRow;
    for (int i = 0; i < argIndices_.size(); i++) {
      // Without the following call to `prepareForReuse`, if the type of
      // `argVectors_[i]` is VARCHAR, then the string buffers will accumulate
      // across calculations. As a result, memory consumption will increase over
      // time. So we call `prepareForReuse` to clear string buffers timely.
      argVectors_[i]->prepareForReuse();
      argVectors_[i]->resize(numFrameRows);
      // Only non-constant field argument vectors need to be populated. The
      // constant vectors are correctly set during aggregate initialization
      // itself.
      if (argIndices_[i] != kConstantChannel) {
        partition_->extractColumn(
            argIndices_[i], firstRow, numFrameRows, 0, argVectors_[i]);
      }
    }
  }

  void computeAggregate(
      SelectivityVector rows,
      vector_size_t startFrame,
      vector_size_t endFrame) {
    rows.clearAll();
    rows.setValidRange(startFrame, endFrame, true);
    rows.updateBounds();

    BaseVector::prepareForReuse(aggregateResultVector_, 1);

    aggregate_->addSingleGroupRawInput(
        rawSingleGroupRow_, rows, argVectors_, false);
    aggregate_->extractValues(&rawSingleGroupRow_, 1, &aggregateResultVector_);
  }

  void incrementalAggregation(
      const SelectivityVector& validRows,
      vector_size_t startFrame,
      vector_size_t endFrame,
      const vector_size_t* rawFrameEnds,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    SelectivityVector rows;
    rows.resize(endFrame + 1 - startFrame);

    auto prevFrameEnd = 0;
    // This is a simple optimization for frames that have a fixed startFrame
    // and increasing frameEnd values. In that case, we can
    // incrementally aggregate over the new rows seen in the frame between
    // the previous and current row.
    validRows.applyToSelected([&](auto i) {
      auto currentFrameEnd = rawFrameEnds[i] - startFrame + 1;
      if (currentFrameEnd > prevFrameEnd) {
        computeAggregate(rows, prevFrameEnd, currentFrameEnd);
      }

      result->copy(aggregateResultVector_.get(), resultOffset + i, 0, 1);
      prevFrameEnd = currentFrameEnd;
    });

    // Set null values for empty (non valid) frames in the output block.
    setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
  }

  /// Implements the two-stack sliding window aggregation for frames where both
  /// frameStart and frameEnd are monotonically non-decreasing.
  void slidingWindowAggregation(
      const SelectivityVector& validRows,
      const FrameMetadata& frameMetadata,
      const vector_size_t* rawFrameStarts,
      const vector_size_t* rawFrameEnds,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    if (!slidingWindowAgg_) {
      slidingWindowAgg_ = std::make_unique<SlidingWindowAgg>(
          aggregate_.get(),
          partialAggregate_.get(),
          singleGroupRowSize_,
          pool_,
          intermediateType_);
    }

    if (!frameMetadata.usePreviousSlidingWindow) {
      // Start a fresh two-stack computation from the first valid frame.
      slidingWindowAgg_->reset();
      auto firstValidRow = validRows.begin();
      leftPtr_ = rawFrameStarts[firstValidRow];
      rightPtr_ = leftPtr_;
    }

    validRows.applyToSelected([&](auto i) {
      auto frameStart = rawFrameStarts[i];
      auto frameEnd = rawFrameEnds[i];

      // Advance right: push rows into the back stack up to frameEnd.
      // Load one row at a time into position 0 of argVectors_ so pushBack
      // can always use a fixed size-1 SelectivityVector instead of a bitmap
      // that grows with each row's absolute position in the batch (O(B²)).
      while (rightPtr_ <= frameEnd) {
        fillArgVectors(rightPtr_, rightPtr_);
        slidingWindowAgg_->pushBack(argVectors_);
        ++rightPtr_;
      }

      // Advance left: pop rows from the front stack until leftPtr_ reaches
      // frameStart.
      while (leftPtr_ < frameStart) {
        slidingWindowAgg_->popFront();
        ++leftPtr_;
      }

      slidingWindowAgg_->extractResult(
          aggregateResultVector_, result, resultOffset + i);
    });

    // Set null values for empty (non valid) frames in the output block.
    setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
  }

  void simpleAggregation(
      const SelectivityVector& validRows,
      vector_size_t minFrame,
      vector_size_t maxFrame,
      const vector_size_t* frameStartsVector,
      const vector_size_t* frameEndsVector,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    SelectivityVector rows;
    rows.resize(maxFrame + 1 - minFrame);
    static auto kSingleGroup = std::vector<vector_size_t>{0};

    validRows.applyToSelected([&](auto i) {
      // Recomputes the entire aggregation for each row by iterating over
      // input rows from frameStart to frameEnd in the SelectivityVector.
      aggregate_->destroy(folly::Range<char**>(&rawSingleGroupRow_, 1));
      aggregate_->initializeNewGroups(&rawSingleGroupRow_, kSingleGroup);
      aggregateInitialized_ = true;

      auto frameStartIndex = frameStartsVector[i] - minFrame;
      auto frameEndIndex = frameEndsVector[i] - minFrame + 1;
      computeAggregate(rows, frameStartIndex, frameEndIndex);
      result->copy(aggregateResultVector_.get(), resultOffset + i, 0, 1);
    });

    // Set null values for empty (non valid) frames in the output block.
    setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
  }

  // Precompute and save the aggregate output for empty input in emptyResult_.
  // This value is returned for rows with empty frames.
  void computeDefaultAggregateValue(const TypePtr& resultType) {
    aggregate_->clear();
    aggregate_->initializeNewGroups(
        &rawSingleGroupRow_, std::vector<vector_size_t>{0});
    aggregateInitialized_ = true;

    emptyResult_ = BaseVector::create(resultType, 1, pool_);
    aggregate_->extractValues(&rawSingleGroupRow_, 1, &emptyResult_);
    aggregate_->clear();
  }

  /// Aggregate function object required for this window function evaluation.
  std::unique_ptr<exec::Aggregate> aggregate_;

  /// kPartial step aggregate sharing the same accumulator layout as aggregate_.
  /// Used by SlidingWindowAgg to extract intermediate accumulator state for
  /// two-stack merging.
  std::unique_ptr<exec::Aggregate> partialAggregate_;

  /// Resolved intermediate (partial) result type for this aggregate function.
  /// Differs from the final resultType_ for functions like avg (ROW vs DOUBLE).
  TypePtr intermediateType_;

  bool aggregateInitialized_{false};

  /// Current WindowPartition used for accessing rows in the apply method.
  const exec::WindowPartition* partition_;

  // Args information : their types, column indexes in inputs and vectors
  // used to populate values to pass to the aggregate function.
  // For a constant argument a column index of kConstantChannel is used in
  // argIndices_, and its ConstantVector value from the Window operator
  // is saved in argVectors_.
  std::vector<TypePtr> argTypes_;
  std::vector<column_index_t> argIndices_;
  std::vector<VectorPtr> argVectors_;

  // This is a single aggregate row needed by the aggregate function for its
  // computation. These values are for the row and its various components.
  BufferPtr singleGroupRowBufferPtr_;
  char* rawSingleGroupRow_;
  vector_size_t singleGroupRowSize_;

  // Used for per-row aggregate computations.
  // This vector is used to copy from the aggregate to the result.
  VectorPtr aggregateResultVector_;

  // Stores metadata about the previous output block of the partition
  // to optimize aggregate computation and reading argument vectors.
  std::optional<FrameMetadata> previousFrameMetadata_;

  // Stores default result value for empty frame aggregation. Window functions
  // return the default value of an aggregate (aggregation with no rows) for
  // empty frames. e.g. count for empty frames should return 0 and not null.
  VectorPtr emptyResult_;

  /// Two-stack state for the sliding window path.
  std::unique_ptr<SlidingWindowAgg> slidingWindowAgg_;
  /// Next partition row index to push into the back stack.
  vector_size_t rightPtr_{0};
  /// Leftmost partition row index currently in the two-stack.
  vector_size_t leftPtr_{0};

  /// When true, skip the two-stack path and fall through to simpleAggregation
  /// even for monotonically sliding frames. Controlled by the
  /// window_sliding_agg_disabled query config; useful for benchmarking and
  /// debugging.
  bool slidingAggDisabled_{false};

  /// Minimum average frame width (in rows) required before activating the
  /// two-stack path for this partition. Set once in the constructor based on
  /// whether the intermediate accumulator type matches the result type:
  ///   - same type (sum, count, min, max): merge is a single arithmetic op,
  ///     breakeven is ~K=400 → threshold = 400.
  ///   - different type (avg → ROW(sum,count)): merge involves ROW
  ///     serialization, breakeven is ~K=1000 → threshold = 1000.
  int32_t slidingWindowMinFrameWidth_{400};

  /// Set on the first apply() call of each partition when the frame is
  /// detected as sliding. Caches whether the two-stack should be used for
  /// the rest of this partition (avoids re-measuring on every batch).
  std::optional<bool> partitionUseTwoStack_;
};

} // namespace

void registerAggregateWindowFunction(const std::string& name) {
  auto aggregateFunctionSignatures = exec::getAggregateFunctionSignatures(name);
  if (aggregateFunctionSignatures.has_value()) {
    // This copy is needed to obtain a vector of the base FunctionSignaturePtr
    // from the AggregateFunctionSignaturePtr type of
    // aggregateFunctionSignatures variable.
    std::vector<exec::FunctionSignaturePtr> signatures(
        aggregateFunctionSignatures.value().begin(),
        aggregateFunctionSignatures.value().end());

    exec::registerWindowFunction(
        name,
        std::move(signatures),
        {exec::WindowFunction::ProcessMode::kRows, true},
        [name](
            const std::vector<exec::WindowFunctionArg>& args,
            const TypePtr& resultType,
            bool ignoreNulls,
            velox::memory::MemoryPool* pool,
            HashStringAllocator* stringAllocator,
            const core::QueryConfig& config)
            -> std::unique_ptr<exec::WindowFunction> {
          return std::make_unique<AggregateWindowFunction>(
              name,
              args,
              resultType,
              ignoreNulls,
              pool,
              stringAllocator,
              config);
        });
  }
}
} // namespace facebook::velox::exec::window
