// Minimal JSON helpers (escaping + number formatting). The result schema is flat
// and fixed, so a full JSON library would be unnecessary weight.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace eib {

inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

inline std::string json_string(const std::string& s) { return "\"" + json_escape(s) + "\""; }

// JSON has no NaN/Inf; emit null so downstream parsers never choke.
inline std::string json_number(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6g", v);
  return buf;
}

}  // namespace eib
