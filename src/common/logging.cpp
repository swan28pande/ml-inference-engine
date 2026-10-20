#include "mie/common/logging.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace mie {
namespace {

std::atomic<LogLevel> g_level{LogLevel::INFO};
std::mutex g_log_mutex;

const char* LevelTag(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG: return "D";
    case LogLevel::INFO: return "I";
    case LogLevel::WARNING: return "W";
    case LogLevel::ERROR: return "E";
  }
  return "?";
}

const char* Basename(const char* path) {
  const char* slash = std::strrchr(path, '/');
  return slash ? slash + 1 : path;
}

}  // namespace

void SetLogLevel(LogLevel level) { g_level.store(level, std::memory_order_relaxed); }
LogLevel GetLogLevel() { return g_level.load(std::memory_order_relaxed); }

bool ParseLogLevel(const char* name, LogLevel* level) {
  std::string n(name ? name : "");
  for (auto& c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (n == "debug") *level = LogLevel::DEBUG;
  else if (n == "info") *level = LogLevel::INFO;
  else if (n == "warning" || n == "warn") *level = LogLevel::WARNING;
  else if (n == "error") *level = LogLevel::ERROR;
  else return false;
  return true;
}

LogMessage::LogMessage(LogLevel level, const char* file, int line) : level_(level) {
  const auto now = std::chrono::system_clock::now();
  const auto t = std::chrono::system_clock::to_time_t(now);
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  stream_ << LevelTag(level) << std::put_time(&tm, "%m%d %H:%M:%S") << '.' << std::setw(6) << std::setfill('0')
          << us << ' ' << std::this_thread::get_id() << ' ' << Basename(file) << ':' << line << "] ";
}

LogMessage::~LogMessage() {
  stream_ << '\n';
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::cerr << stream_.str();
  if (level_ >= LogLevel::WARNING) std::cerr.flush();
}

}  // namespace mie
