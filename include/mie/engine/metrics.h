// Lock-free latency histograms, throughput meters and per-model metrics.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mie/common/clock.h"

namespace mie {

// Log-linear histogram over non-negative integer values (microseconds).
// Values below 16 are recorded exactly; above that each power of two is split
// into 16 sub-buckets, bounding relative error to ~6%. Record() is wait-free.
class LatencyHistogram {
 public:
  static constexpr int kSubBucketBits = 4;
  static constexpr int kSubBuckets = 1 << kSubBucketBits;
  static constexpr int kMaxExponent = 40;  // ~12.7 days in microseconds
  static constexpr int kNumBuckets = kSubBuckets + (kMaxExponent - kSubBucketBits + 1) * kSubBuckets;

  struct Summary {
    uint64_t count = 0;
    double mean = 0;
    double p50 = 0;
    double p95 = 0;
    double p99 = 0;
    double p999 = 0;
    int64_t min = 0;
    int64_t max = 0;
  };

  LatencyHistogram();

  void Record(int64_t value);
  // Percentile in [0, 100]. Returns 0 when empty.
  double Percentile(double p) const;
  Summary Summarize() const;
  uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  void Reset();
  void Merge(const LatencyHistogram& other);

  static int BucketIndex(int64_t value);
  static int64_t BucketLowerBound(int index);
  static int64_t BucketUpperBound(int index);  // exclusive

 private:
  double PercentileFromCounts(const std::vector<uint64_t>& counts, uint64_t total, double p) const;

  std::array<std::atomic<uint64_t>, kNumBuckets> buckets_;
  std::atomic<uint64_t> count_{0};
  std::atomic<int64_t> sum_{0};
  std::atomic<int64_t> min_;
  std::atomic<int64_t> max_{0};
};

// Events-per-second over a sliding window of whole seconds.
class ThroughputMeter {
 public:
  static constexpr int kWindowSlots = 64;
  ThroughputMeter();
  void Record(uint64_t events = 1);
  // Average rate over the last `window_seconds` complete seconds.
  double Rate(int window_seconds = 10) const;
  uint64_t total() const { return total_.load(std::memory_order_relaxed); }

 private:
  int64_t NowSeconds() const;
  struct Slot {
    std::atomic<int64_t> second{-1};
    std::atomic<uint64_t> count{0};
  };
  TimePoint start_;
  mutable std::mutex mu_;  // guards slot rollover only
  std::array<Slot, kWindowSlots> slots_;
  std::atomic<uint64_t> total_{0};
};

struct ModelMetricsSnapshot {
  std::string model_name;
  std::string version;
  uint64_t requests_ok = 0;
  uint64_t requests_failed = 0;
  uint64_t requests_rejected = 0;
  uint64_t batches = 0;
  double throughput_rps = 0;
  double mean_batch_size = 0;
  LatencyHistogram::Summary request_latency;
  LatencyHistogram::Summary queue_latency;
  LatencyHistogram::Summary compute_latency;
  LatencyHistogram::Summary batch_size;
};

// Metrics for a single model version, updated by the scheduler.
class ModelMetrics {
 public:
  ModelMetrics(std::string model_name, std::string version);

  void RecordSuccess(int64_t total_us, int64_t queue_us);
  void RecordFailure();
  void RecordRejected();
  void RecordBatch(int batch_size, int64_t compute_us);

  ModelMetricsSnapshot Snapshot() const;
  const std::string& model_name() const { return model_name_; }
  const std::string& version() const { return version_; }

 private:
  std::string model_name_;
  std::string version_;
  std::atomic<uint64_t> ok_{0};
  std::atomic<uint64_t> failed_{0};
  std::atomic<uint64_t> rejected_{0};
  std::atomic<uint64_t> batches_{0};
  std::atomic<uint64_t> batched_samples_{0};
  LatencyHistogram request_latency_;
  LatencyHistogram queue_latency_;
  LatencyHistogram compute_latency_;
  LatencyHistogram batch_size_;
  ThroughputMeter throughput_;
};

// Registry of ModelMetrics keyed by (model, version). Thread-safe.
class MetricsRegistry {
 public:
  std::shared_ptr<ModelMetrics> GetOrCreate(const std::string& model, const std::string& version);
  std::vector<ModelMetricsSnapshot> SnapshotAll() const;
  // Prometheus text exposition format (counters + summary quantiles).
  std::string RenderPrometheus() const;

 private:
  mutable std::mutex mu_;
  std::map<std::pair<std::string, std::string>, std::shared_ptr<ModelMetrics>> metrics_;
};

}  // namespace mie
