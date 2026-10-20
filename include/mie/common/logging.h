// Minimal thread-safe, level-filtered logger.
//
//   MIE_LOG(INFO) << "loaded model " << name;
#pragma once

#include <sstream>

namespace mie {

enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };

void SetLogLevel(LogLevel level);
LogLevel GetLogLevel();
// Parses "debug" / "info" / "warning" / "error" (case-insensitive).
bool ParseLogLevel(const char* name, LogLevel* level);

class LogMessage {
 public:
  LogMessage(LogLevel level, const char* file, int line);
  ~LogMessage();
  LogMessage(const LogMessage&) = delete;
  LogMessage& operator=(const LogMessage&) = delete;
  std::ostringstream& stream() { return stream_; }

 private:
  LogLevel level_;
  std::ostringstream stream_;
};

// Lets MIE_LOG expand to a single expression (ternary), so it is safe inside
// an unbraced if/else.
class LogMessageVoidify {
 public:
  void operator&(std::ostream&) {}
};

}  // namespace mie

#define MIE_LOG(level)                                    \
  (::mie::LogLevel::level < ::mie::GetLogLevel())         \
      ? (void)0                                           \
      : ::mie::LogMessageVoidify() &                      \
            ::mie::LogMessage(::mie::LogLevel::level, __FILE__, __LINE__).stream()
