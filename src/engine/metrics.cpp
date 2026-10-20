#include "mie/engine/metrics.h"

#include <algorithm>
#include <limits>
#include <sstream>

namespace mie {
namespace {

int HighestBit(uint64_t v) {
  int n = -1;
  while (v) {
    v >>= 1;
    ++n;
  }
  return n;
}

}  // namespace

// ------------------------------------------------------------ LatencyHistogram

LatencyHistogram::LatencyHistogram() : min_(std::numeric_limits<int64_t>::max()) {
  for (auto& b : buckets_) b.store(0, std::memory_order_relaxed);
}

int LatencyHistogram::BucketIndex(int64_t value) {
  if (value < 0) value = 0;
  if (value < kSubBuckets) return static_cast<int>(value);
  int exp = HighestBit(static_cast<uint64_t>(value));
  if (exp > kMaxExponent) return kNumBuckets - 1;
  const int shift = exp - kSubBucketBits;
  const int sub = static_cast<int>((static_cast<uint64_t>(value) >> shift) - kSubBuckets);
  return kSubBuckets + (exp - kSubBucketBits) * kSubBuckets + sub;
}

int64_t LatencyHistogram::BucketLowerBound(int index) {
  if (index < kSubBuckets) return index;
  const int group = (index - kSubBuckets) / kSubBuckets;  // exp - kSubBucketBits
  const int sub = (index - kSubBuckets) % kSubBuckets;
  return static_cast<int64_t>(kSubBuckets + sub) << group;
}

int64_t LatencyHistogram::BucketUpperBound(int index) {
  if (index < kSubBuckets) return index + 1;
  const int group = (index - kSubBuckets) / kSubBuckets;
  const int sub = (index - kSubBuckets) % kSubBuckets;
  return static_cast<int64_t>(kSubBuckets + sub + 1) << group;
}

void LatencyHistogram::Record(int64_t value) {
  if (value < 0) value = 0;
  buckets_[static_cast<size_t>(BucketIndex(value))].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  sum_.fetch_add(value, std::memory_order_relaxed);
  int64_t cur = min_.load(std::memory_order_relaxed);
  while (value < cur && !min_.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
  }
  cur = max_.load(std::memory_order_relaxed);
  while (value > cur && !max_.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
  }
}

double LatencyHistogram::PercentileFromCounts(const std::vector<uint64_t>& counts, uint64_t total,
                                              double p) const {
  if (total == 0) return 0.0;
  p = std::clamp(p, 0.0, 100.0);
  // Rank of the target sample (1-based, nearest-rank definition).
  const auto rank = std::max<uint64_t>(1, static_cast<uint64_t>(p / 100.0 * static_cast<double>(total) + 0.5));
  uint64_t cumulative = 0;
  for (int i = 0; i < kNumBuckets; ++i) {
    const uint64_t c = counts[static_cast<size_t>(i)];
    if (c == 0) continue;
    if (cumulative + c >= rank) {
      // Linear interpolation inside the bucket.
      const double lo = static_cast<double>(BucketLowerBound(i));
      const double hi = static_cast<double>(BucketUpperBound(i));
      const double frac = static_cast<double>(rank - cumulative) / static_cast<double>(c);
      double v = lo + (hi - lo) * frac;
      if (i < kSubBuckets) v = lo;  // exact buckets
      const double mn = static_cast<double>(min_.load(std::memory_order_relaxed));
      const double mx = static_cast<double>(max_.load(std::memory_order_relaxed));
      return std::clamp(v, mn, mx);
    }
    cumulative += c;
  }
  return static_cast<double>(max_.load(std::memory_order_relaxed));
}

double LatencyHistogram::Percentile(double p) const {
  std::vector<uint64_t> counts(kNumBuckets);
  uint64_t total = 0;
  for (int i = 0; i < kNumBuckets; ++i) {
    counts[static_cast<size_t>(i)] = buckets_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
    total += counts[static_cast<size_t>(i)];
  }
  return PercentileFromCounts(counts, total, p);
}

LatencyHistogram::Summary LatencyHistogram::Summarize() const {
  std::vector<uint64_t> counts(kNumBuckets);
  uint64_t total = 0;
  for (int i = 0; i < kNumBuckets; ++i) {
    counts[static_cast<size_t>(i)] = buckets_[static_cast<size_t>(i)].load(std::memory_order_relaxed);
    total += counts[static_cast<size_t>(i)];
  }
  Summary s;
  s.count = total;
  if (total == 0) return s;
  s.mean = static_cast<double>(sum_.load(std::memory_order_relaxed)) / static_cast<double>(count_.load());
  s.p50 = PercentileFromCounts(counts, total, 50);
  s.p95 = PercentileFromCounts(counts, total, 95);
  s.p99 = PercentileFromCounts(counts, total, 99);
  s.p999 = PercentileFromCounts(counts, total, 99.9);
  s.min = min_.load(std::memory_order_relaxed);
  s.max = max_.load(std::memory_order_relaxed);
  return s;
}

void LatencyHistogram::Reset() {
  for (auto& b : buckets_) b.store(0, std::memory_order_relaxed);
  count_.store(0);
  sum_.store(0);
  min_.store(std::numeric_limits<int64_t>::max());
  max_.store(0);
}

void LatencyHistogram::Merge(const LatencyHistogram& other) {
  for (int i = 0; i < kNumBuckets; ++i) {
    buckets_[static_cast<size_t>(i)].fetch_add(other.buckets_[static_cast<size_t>(i)].load(),
                                                std::memory_order_relaxed);
  }
  count_.fetch_add(other.count_.load());
  sum_.fetch_add(other.sum_.load());
  const int64_t omin = other.min_.load();
  int64_t cur = min_.load();
  while (omin < cur && !min_.compare_exchange_weak(cur, omin)) {
  }
  const int64_t omax = other.max_.load();
  cur = max_.load();
  while (omax > cur && !max_.compare_exchange_weak(cur, omax)) {
  }
}

// ------------------------------------------------------------- ThroughputMeter

ThroughputMeter::ThroughputMeter() : start_(Clock::now()) {}

int64_t ThroughputMeter::NowSeconds() const {
  return std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start_).count();
}

void ThroughputMeter::Record(uint64_t events) {
  total_.fetch_add(events, std::memory_order_relaxed);
  const int64_t sec = NowSeconds();
  Slot& slot = slots_[static_cast<size_t>(sec % kWindowSlots)];
  if (slot.second.load(std::memory_order_acquire) != sec) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot.second.load(std::memory_order_relaxed) != sec) {
      slot.count.store(0, std::memory_order_relaxed);
      slot.second.store(sec, std::memory_order_release);
    }
  }
  slot.count.fetch_add(events, std::memory_order_relaxed);
}

double ThroughputMeter::Rate(int window_seconds) const {
  window_seconds = std::clamp(window_seconds, 1, kWindowSlots - 1);
  const int64_t now = NowSeconds();
  // Only count complete seconds; if the meter is younger than the window,
  // average over the elapsed time instead.
  const int64_t window = std::min<int64_t>(window_seconds, now);
  if (window <= 0) {
    const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
    return elapsed > 0 ? static_cast<double>(total()) / elapsed : 0.0;
  }
  uint64_t sum = 0;
  for (int64_t s = now - window; s < now; ++s) {
    const Slot& slot = slots_[static_cast<size_t>(s % kWindowSlots)];
    if (slot.second.load(std::memory_order_acquire) == s) sum += slot.count.load(std::memory_order_relaxed);
  }
  return static_cast<double>(sum) / static_cast<double>(window);
}

// ----------------------------------------------------------------- ModelMetrics

ModelMetrics::ModelMetrics(std::string model_name, std::string version)
    : model_name_(std::move(model_name)), version_(std::move(version)) {}

void ModelMetrics::RecordSuccess(int64_t total_us, int64_t queue_us) {
  ok_.fetch_add(1, std::memory_order_relaxed);
  request_latency_.Record(total_us);
  queue_latency_.Record(queue_us);
  throughput_.Record();
}

void ModelMetrics::RecordFailure() { failed_.fetch_add(1, std::memory_order_relaxed); }
void ModelMetrics::RecordRejected() { rejected_.fetch_add(1, std::memory_order_relaxed); }

void ModelMetrics::RecordBatch(int batch_size, int64_t compute_us) {
  batches_.fetch_add(1, std::memory_order_relaxed);
  batched_samples_.fetch_add(static_cast<uint64_t>(batch_size), std::memory_order_relaxed);
  compute_latency_.Record(compute_us);
  batch_size_.Record(batch_size);
}

ModelMetricsSnapshot ModelMetrics::Snapshot() const {
  ModelMetricsSnapshot s;
  s.model_name = model_name_;
  s.version = version_;
  s.requests_ok = ok_.load();
  s.requests_failed = failed_.load();
  s.requests_rejected = rejected_.load();
  s.batches = batches_.load();
  s.throughput_rps = throughput_.Rate(10);
  s.mean_batch_size = s.batches ? static_cast<double>(batched_samples_.load()) / static_cast<double>(s.batches) : 0;
  s.request_latency = request_latency_.Summarize();
  s.queue_latency = queue_latency_.Summarize();
  s.compute_latency = compute_latency_.Summarize();
  s.batch_size = batch_size_.Summarize();
  return s;
}

// -------------------------------------------------------------- MetricsRegistry

std::shared_ptr<ModelMetrics> MetricsRegistry::GetOrCreate(const std::string& model, const std::string& version) {
  std::lock_guard<std::mutex> lock(mu_);
  auto& slot = metrics_[{model, version}];
  if (!slot) slot = std::make_shared<ModelMetrics>(model, version);
  return slot;
}

std::vector<ModelMetricsSnapshot> MetricsRegistry::SnapshotAll() const {
  std::vector<std::shared_ptr<ModelMetrics>> all;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [key, m] : metrics_) all.push_back(m);
  }
  std::vector<ModelMetricsSnapshot> out;
  out.reserve(all.size());
  for (const auto& m : all) out.push_back(m->Snapshot());
  return out;
}

std::string MetricsRegistry::RenderPrometheus() const {
  const auto snapshots = SnapshotAll();
  std::ostringstream os;
  auto labels = [](const ModelMetricsSnapshot& s) {
    return "model=\"" + s.model_name + "\",version=\"" + s.version + "\"";
  };
  auto summary = [&](const char* name, const char* help, const ModelMetricsSnapshot& s,
                     const LatencyHistogram::Summary& h) {
    const std::string l = labels(s);
    os << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " summary\n";
    os << name << '{' << l << ",quantile=\"0.5\"} " << h.p50 << '\n';
    os << name << '{' << l << ",quantile=\"0.95\"} " << h.p95 << '\n';
    os << name << '{' << l << ",quantile=\"0.99\"} " << h.p99 << '\n';
    os << name << "_count{" << l << "} " << h.count << '\n';
  };
  for (const auto& s : snapshots) {
    const std::string l = labels(s);
    os << "# TYPE mie_requests_total counter\n";
    os << "mie_requests_total{" << l << ",status=\"ok\"} " << s.requests_ok << '\n';
    os << "mie_requests_total{" << l << ",status=\"failed\"} " << s.requests_failed << '\n';
    os << "mie_requests_total{" << l << ",status=\"rejected\"} " << s.requests_rejected << '\n';
    os << "# TYPE mie_throughput_rps gauge\nmie_throughput_rps{" << l << "} " << s.throughput_rps << '\n';
    os << "# TYPE mie_batch_size_mean gauge\nmie_batch_size_mean{" << l << "} " << s.mean_batch_size << '\n';
    summary("mie_request_latency_us", "End-to-end engine latency in microseconds", s, s.request_latency);
    summary("mie_queue_latency_us", "Time spent waiting for a batch in microseconds", s, s.queue_latency);
    summary("mie_compute_latency_us", "Backend batch execution time in microseconds", s, s.compute_latency);
  }
  return os.str();
}

}  // namespace mie
