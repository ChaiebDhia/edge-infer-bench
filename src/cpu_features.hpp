// Parses /proc/cpuinfo text so every benchmark result records which SIMD and
// ML-relevant instruction extensions the machine actually exposes. Taking the
// text as an argument (instead of reading the file inside) keeps it unit-testable.
#pragma once

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace eib {

struct CpuInfo {
  std::string model;                  // "model name" on x86, implementer/part on Arm
  std::vector<std::string> features;  // ML-relevant subset, in a stable order
};

inline std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Extensions that matter for NN inference. asimd = Arm NEON, asimddp = dot-product
// (SDOT/UDOT), asimdhp/fphp = FP16 arithmetic, i8mm = int8 matrix multiply,
// bf16 = bfloat16, sve/sve2/sme = scalable vector/matrix extensions.
inline const std::vector<std::string>& interesting_features() {
  static const std::vector<std::string> k = {"asimd",   "asimddp",  "asimdhp",     "fphp",     "i8mm",    "bf16",
                                             "sve",     "sve2",     "sme",         "avx",      "avx2",    "fma",
                                             "avx512f", "avx512bw", "avx512_vnni", "avx_vnni", "amx_int8"};
  return k;
}

inline CpuInfo parse_cpuinfo(const std::string& text) {
  CpuInfo info;
  std::string implementer, part, flags_line;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon));
    const std::string value = trim(line.substr(colon + 1));
    if (key == "model name" && info.model.empty())
      info.model = value;
    else if (key == "CPU implementer" && implementer.empty())
      implementer = value;
    else if (key == "CPU part" && part.empty())
      part = value;
    else if ((key == "Features" || key == "flags") && flags_line.empty())
      flags_line = value;
  }
  if (info.model.empty() && !implementer.empty()) {
    info.model = "arm implementer=" + implementer + " part=" + part;
  }
  std::istringstream fs(flags_line);
  const std::vector<std::string> present{std::istream_iterator<std::string>(fs), std::istream_iterator<std::string>()};
  for (const auto& f : interesting_features()) {
    if (std::find(present.begin(), present.end(), f) != present.end()) info.features.push_back(f);
  }
  return info;
}

inline CpuInfo read_cpuinfo(const std::string& path = "/proc/cpuinfo") {
  std::ifstream f(path);
  if (!f) return CpuInfo{};
  std::stringstream ss;
  ss << f.rdbuf();
  return parse_cpuinfo(ss.str());
}

}  // namespace eib
