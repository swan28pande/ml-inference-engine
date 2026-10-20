#include "mie/common/status.h"

namespace mie {

const char* StatusCodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "OK";
    case StatusCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case StatusCode::kNotFound: return "NOT_FOUND";
    case StatusCode::kAlreadyExists: return "ALREADY_EXISTS";
    case StatusCode::kFailedPrecondition: return "FAILED_PRECONDITION";
    case StatusCode::kResourceExhausted: return "RESOURCE_EXHAUSTED";
    case StatusCode::kUnavailable: return "UNAVAILABLE";
    case StatusCode::kDeadlineExceeded: return "DEADLINE_EXCEEDED";
    case StatusCode::kCancelled: return "CANCELLED";
    case StatusCode::kUnimplemented: return "UNIMPLEMENTED";
    case StatusCode::kInternal: return "INTERNAL";
  }
  return "UNKNOWN";
}

std::string Status::ToString() const {
  if (ok()) return "OK";
  return std::string(StatusCodeName(code_)) + ": " + message_;
}

}  // namespace mie
