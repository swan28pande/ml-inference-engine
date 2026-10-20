// Shared helpers for the benchmark binaries: flag parsing and report output.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mie/engine/metrics.h"

namespace mie {
namespace bench {

// Parses --key=value and --key value flags.
class Flags {
 public:
  Flags(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("--", 0) != 0) continue;
      arg = arg.substr(2);
      const auto eq = arg.find('=');
      if (eq != std::string::npos) {
        values_[arg.substr(0, eq)] = arg.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        values_[arg] = argv[++i];
      } else {
        values_[arg] = "true";
      }
    }
  }
  bool Has(const std::string& k) const { return values_.count(k) > 0; }
  std::string Str(const std::string& k, const std::string& d) const {
    auto it = values_.find(k);
    return it == values_.end() ? d : it->second;
  }
  int64_t Int(const std::string& k, int64_t d) const { return Has(k) ? std::stoll(values_.at(k)) : d; }
  double Double(const std::string& k, double d) const { return Has(k) ? std::stod(values_.at(k)) : d; }
  std::vector<int> IntList(const std::string& k, const std::string& d) const {
    std::vector<int> out;
    std::stringstream ss(Str(k, d));
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
  }

 private:
  std::map<std::string, std::string> values_;
};

inline std::string FormatMs(double micros) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f", micros / 1000.0);
  return buf;
}

inline std::string FormatRate(double rps) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.0f", rps);
  return buf;
}

}  // namespace bench
}  // namespace mie
