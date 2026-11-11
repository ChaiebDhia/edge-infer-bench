// Small, dependency-free latency statistics used by the benchmark harness.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace eib {

// Linear-interpolated percentile (matches numpy.percentile's default method).
inline double percentile(std::vector<double> values, double p) {
  if (values.empty()) throw std::invalid_argument("percentile: empty input");
  if (p < 0.0 || p > 100.0) throw std::invalid_argument("percentile: p must be in [0, 100]");
  std::sort(values.begin(), values.end());
  const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(rank));
  const auto hi = static_cast<std::size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return values[lo] + (values[hi] - values[lo]) * frac;
}

struct Summary {
  double mean = 0.0;
  double stddev = 0.0;  // sample standard deviation (n - 1)
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
};

inline Summary summarize(const std::vector<double>& values) {
  if (values.empty()) throw std::invalid_argument("summarize: empty input");
  Summary s;
  const auto n = static_cast<double>(values.size());
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
  double sq = 0.0;
  for (double v : values) sq += (v - s.mean) * (v - s.mean);
  s.stddev = values.size() > 1 ? std::sqrt(sq / (n - 1.0)) : 0.0;
  const auto [mn, mx] = std::minmax_element(values.begin(), values.end());
  s.min = *mn;
  s.max = *mx;
  s.p50 = percentile(values, 50.0);
  s.p95 = percentile(values, 95.0);
  s.p99 = percentile(values, 99.0);
  return s;
}

}  // namespace eib
