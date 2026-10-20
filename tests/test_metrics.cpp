#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <thread>
#include <vector>

#include "mie/engine/metrics.h"

namespace mie {
namespace {

TEST(LatencyHistogramTest, BucketBoundsAreConsistent) {
  for (int64_t v : {0, 1, 15, 16, 17, 31, 32, 100, 999, 1000, 123456, 10'000'000}) {
    const int idx = LatencyHistogram::BucketIndex(v);
    EXPECT_LE(LatencyHistogram::BucketLowerBound(idx), v) << v;
    EXPECT_GT(LatencyHistogram::BucketUpperBound(idx), v) << v;
  }
  for (int i = 1; i < LatencyHistogram::kNumBuckets; ++i) {
    EXPECT_EQ(LatencyHistogram::BucketLowerBound(i), LatencyHistogram::BucketUpperBound(i - 1));
  }
}

TEST(LatencyHistogramTest, EmptyHistogram) {
  LatencyHistogram h;
  EXPECT_EQ(h.count(), 0u);
  EXPECT_EQ(h.Percentile(99), 0.0);
  EXPECT_EQ(h.Summarize().count, 0u);
}

TEST(LatencyHistogramTest, ExactForSmallValues) {
  LatencyHistogram h;
  for (int i = 0; i < 10; ++i) h.Record(i);
  EXPECT_EQ(h.Percentile(50), 4.0);
  EXPECT_EQ(h.Summarize().min, 0);
  EXPECT_EQ(h.Summarize().max, 9);
}

TEST(LatencyHistogramTest, PercentilesWithinRelativeError) {
  std::mt19937 rng(7);
  std::lognormal_distribution<double> dist(7.0, 0.8);  // ~1ms median, long tail
  std::vector<int64_t> values(100000);
  LatencyHistogram h;
  for (auto& v : values) {
    v = static_cast<int64_t>(dist(rng));
    h.Record(v);
  }
  std::sort(values.begin(), values.end());
  for (double p : {50.0, 90.0, 95.0, 99.0, 99.9}) {
    const double exact = static_cast<double>(values[static_cast<size_t>(p / 100.0 * (values.size() - 1))]);
    EXPECT_NEAR(h.Percentile(p), exact, exact * 0.07) << "p" << p;
  }
  const auto s = h.Summarize();
  EXPECT_EQ(s.count, values.size());
  EXPECT_EQ(s.max, values.back());
  EXPECT_LE(s.p50, s.p95);
  EXPECT_LE(s.p95, s.p99);
}

TEST(LatencyHistogramTest, ConcurrentRecording) {
  LatencyHistogram h;
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 10000; ++i) h.Record(i % 1000);
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(h.count(), 80000u);
  EXPECT_NEAR(h.Percentile(50), 500, 40);
}

TEST(LatencyHistogramTest, MergeAndReset) {
  LatencyHistogram a, b;
  for (int i = 0; i < 100; ++i) a.Record(100);
  for (int i = 0; i < 100; ++i) b.Record(10000);
  a.Merge(b);
  EXPECT_EQ(a.count(), 200u);
  EXPECT_EQ(a.Summarize().max, 10000);
  a.Reset();
  EXPECT_EQ(a.count(), 0u);
}

TEST(ThroughputMeterTest, CountsEvents) {
  ThroughputMeter m;
  for (int i = 0; i < 500; ++i) m.Record();
  EXPECT_EQ(m.total(), 500u);
  EXPECT_GT(m.Rate(), 0.0);  // young meter: averaged over elapsed time
}

TEST(ModelMetricsTest, SnapshotAndPrometheus) {
  MetricsRegistry registry;
  auto m = registry.GetOrCreate("resnet", "2");
  EXPECT_EQ(m, registry.GetOrCreate("resnet", "2"));
  for (int i = 0; i < 10; ++i) m->RecordSuccess(1000 + i, 100);
  m->RecordFailure();
  m->RecordRejected();
  m->RecordBatch(8, 500);
  m->RecordBatch(4, 300);

  const auto snaps = registry.SnapshotAll();
  ASSERT_EQ(snaps.size(), 1u);
  EXPECT_EQ(snaps[0].requests_ok, 10u);
  EXPECT_EQ(snaps[0].requests_failed, 1u);
  EXPECT_EQ(snaps[0].requests_rejected, 1u);
  EXPECT_DOUBLE_EQ(snaps[0].mean_batch_size, 6.0);

  const std::string text = registry.RenderPrometheus();
  EXPECT_NE(text.find("mie_requests_total{model=\"resnet\",version=\"2\",status=\"ok\"} 10"), std::string::npos);
  EXPECT_NE(text.find("mie_request_latency_us{model=\"resnet\",version=\"2\",quantile=\"0.99\"}"), std::string::npos);
}

}  // namespace
}  // namespace mie
