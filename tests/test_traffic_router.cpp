#include <gtest/gtest.h>

#include <map>
#include <string>

#include "mie/server/traffic_router.h"

namespace mie {
namespace {

const std::vector<std::string> kReady = {"1", "2"};

TEST(TrafficRouterTest, ExplicitVersionWins) {
  TrafficRouter router;
  ASSERT_TRUE(router.SetSplit("m", {{"1", 100}}).ok());
  auto v = router.Route("m", "2", "", kReady);
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value(), "2");
}

TEST(TrafficRouterTest, NoSplitRoutesToLatest) {
  TrafficRouter router;
  auto v = router.Route("m", "", "", {"1", "2", "3"});
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value(), "3");
  EXPECT_FALSE(router.Route("m", "", "", {}).ok());
}

TEST(TrafficRouterTest, WeightedSplitMatchesWeights) {
  TrafficRouter router;
  ASSERT_TRUE(router.SetSplit("m", {{"1", 80}, {"2", 20}}).ok());
  std::map<std::string, int> counts;
  constexpr int kN = 50000;
  for (int i = 0; i < kN; ++i) counts[router.Route("m", "", "", kReady).value()]++;
  EXPECT_NEAR(counts["1"] / static_cast<double>(kN), 0.8, 0.02);
  EXPECT_NEAR(counts["2"] / static_cast<double>(kN), 0.2, 0.02);
}

TEST(TrafficRouterTest, StickyRoutingIsDeterministicAndBalanced) {
  TrafficRouter router;
  ASSERT_TRUE(router.SetSplit("m", {{"1", 50}, {"2", 50}}).ok());
  std::map<std::string, int> counts;
  for (int u = 0; u < 20000; ++u) {
    const std::string key = "user-" + std::to_string(u);
    const std::string first = router.Route("m", "", key, kReady).value();
    for (int rep = 0; rep < 3; ++rep) EXPECT_EQ(router.Route("m", "", key, kReady).value(), first);
    counts[first]++;
  }
  EXPECT_NEAR(counts["1"] / 20000.0, 0.5, 0.03);
}

TEST(TrafficRouterTest, UnavailableVersionsAreSkipped) {
  TrafficRouter router;
  ASSERT_TRUE(router.SetSplit("m", {{"1", 50}, {"3", 50}}).ok());
  for (int i = 0; i < 100; ++i) EXPECT_EQ(router.Route("m", "", "", kReady).value(), "1");
  ASSERT_TRUE(router.SetSplit("m", {{"7", 100}}).ok());
  EXPECT_EQ(router.Route("m", "", "", kReady).value(), "2");  // falls back to latest
}

TEST(TrafficRouterTest, ValidatesAndClearsSplits) {
  TrafficRouter router;
  EXPECT_FALSE(router.SetSplit("m", {{"1", 0}, {"2", 0}}).ok());
  EXPECT_FALSE(router.SetSplit("m", {{"", 10}}).ok());
  ASSERT_TRUE(router.SetSplit("m", {{"1", 10}}).ok());
  EXPECT_EQ(router.GetSplit("m").size(), 1u);
  ASSERT_TRUE(router.SetSplit("m", {}).ok());
  EXPECT_TRUE(router.GetSplit("m").empty());
}

}  // namespace
}  // namespace mie
