#pragma once

#include <chrono>
#include <cstdint>

namespace mie {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline int64_t MicrosBetween(TimePoint start, TimePoint end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

inline int64_t MicrosSince(TimePoint start) { return MicrosBetween(start, Clock::now()); }

}  // namespace mie
