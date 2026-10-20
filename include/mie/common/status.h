// Lightweight error handling used across the engine (no exceptions on hot paths).
#pragma once

#include <optional>
#include <string>
#include <utility>

namespace mie {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kFailedPrecondition,
  kResourceExhausted,
  kUnavailable,
  kDeadlineExceeded,
  kCancelled,
  kUnimplemented,
  kInternal,
};

const char* StatusCodeName(StatusCode code);

class [[nodiscard]] Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }
  std::string ToString() const;

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

inline Status InvalidArgument(std::string m) { return {StatusCode::kInvalidArgument, std::move(m)}; }
inline Status NotFound(std::string m) { return {StatusCode::kNotFound, std::move(m)}; }
inline Status AlreadyExists(std::string m) { return {StatusCode::kAlreadyExists, std::move(m)}; }
inline Status FailedPrecondition(std::string m) { return {StatusCode::kFailedPrecondition, std::move(m)}; }
inline Status ResourceExhausted(std::string m) { return {StatusCode::kResourceExhausted, std::move(m)}; }
inline Status Unavailable(std::string m) { return {StatusCode::kUnavailable, std::move(m)}; }
inline Status DeadlineExceeded(std::string m) { return {StatusCode::kDeadlineExceeded, std::move(m)}; }
inline Status Cancelled(std::string m) { return {StatusCode::kCancelled, std::move(m)}; }
inline Status Unimplemented(std::string m) { return {StatusCode::kUnimplemented, std::move(m)}; }
inline Status Internal(std::string m) { return {StatusCode::kInternal, std::move(m)}; }

// Holds either a value or an error Status. Supports move-only T.
template <typename T>
class [[nodiscard]] StatusOr {
 public:
  StatusOr(Status status) : status_(std::move(status)) {}  // NOLINT(implicit)
  StatusOr(T value) : value_(std::move(value)) {}         // NOLINT(implicit)

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  T& value() & { return *value_; }
  const T& value() const& { return *value_; }
  T&& value() && { return std::move(*value_); }

  T& operator*() & { return *value_; }
  T* operator->() { return &*value_; }
  const T* operator->() const { return &*value_; }

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace mie

#define MIE_CONCAT_INNER(a, b) a##b
#define MIE_CONCAT(a, b) MIE_CONCAT_INNER(a, b)

#define MIE_RETURN_IF_ERROR(expr)              \
  do {                                         \
    ::mie::Status _mie_status = (expr);        \
    if (!_mie_status.ok()) return _mie_status; \
  } while (0)

#define MIE_ASSIGN_OR_RETURN(lhs, expr)                                  \
  auto MIE_CONCAT(_mie_or_, __LINE__) = (expr);                          \
  if (!MIE_CONCAT(_mie_or_, __LINE__).ok())                              \
    return MIE_CONCAT(_mie_or_, __LINE__).status();                      \
  lhs = std::move(MIE_CONCAT(_mie_or_, __LINE__)).value()
