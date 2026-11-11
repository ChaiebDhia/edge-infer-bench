#include <gtest/gtest.h>

#include <vector>

#include "cpu_features.hpp"
#include "json.hpp"
#include "stats.hpp"

TEST(Stats, PercentileMatchesNumpyConvention) {
  const std::vector<double> v{1, 2, 3, 4, 5};
  EXPECT_DOUBLE_EQ(eib::percentile(v, 0), 1.0);
  EXPECT_DOUBLE_EQ(eib::percentile(v, 50), 3.0);
  EXPECT_DOUBLE_EQ(eib::percentile(v, 100), 5.0);
  EXPECT_DOUBLE_EQ(eib::percentile(v, 25), 2.0);
  // numpy.percentile([1,2,3,4], 95) == 3.85
  EXPECT_NEAR(eib::percentile({1, 2, 3, 4}, 95), 3.85, 1e-12);
}

TEST(Stats, PercentileIsOrderIndependentAndValidates) {
  EXPECT_DOUBLE_EQ(eib::percentile({5, 1, 4, 2, 3}, 50), 3.0);
  EXPECT_THROW(eib::percentile({}, 50), std::invalid_argument);
  EXPECT_THROW(eib::percentile({1.0}, 101), std::invalid_argument);
}

TEST(Stats, SummarizeBasics) {
  const auto s = eib::summarize({2, 4, 4, 4, 5, 5, 7, 9});
  EXPECT_DOUBLE_EQ(s.mean, 5.0);
  EXPECT_NEAR(s.stddev, 2.13808993529939, 1e-9);  // sample stddev
  EXPECT_DOUBLE_EQ(s.min, 2.0);
  EXPECT_DOUBLE_EQ(s.max, 9.0);
  EXPECT_DOUBLE_EQ(s.p50, 4.5);
}

TEST(Stats, SingleSampleHasZeroStddev) {
  const auto s = eib::summarize({3.5});
  EXPECT_DOUBLE_EQ(s.stddev, 0.0);
  EXPECT_DOUBLE_EQ(s.p99, 3.5);
}

TEST(CpuFeatures, ParsesArmFeaturesLine) {
  const std::string text =
      "processor\t: 0\n"
      "BogoMIPS\t: 2000.00\n"
      "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm "
      "lrcpc dcpop asimddp ssbs sve i8mm bf16\n"
      "CPU implementer\t: 0x41\n"
      "CPU part\t: 0xd40\n";
  const auto info = eib::parse_cpuinfo(text);
  EXPECT_EQ(info.model, "arm implementer=0x41 part=0xd40");
  const std::vector<std::string> expected{"asimd", "asimddp", "asimdhp", "fphp", "i8mm", "bf16", "sve"};
  EXPECT_EQ(info.features, expected);
}

TEST(CpuFeatures, ParsesX86FlagsLine) {
  const std::string text =
      "model name\t: Example CPU @ 2.50GHz\n"
      "flags\t\t: fpu sse sse2 avx avx2 fma avx512f avx512_vnni\n";
  const auto info = eib::parse_cpuinfo(text);
  EXPECT_EQ(info.model, "Example CPU @ 2.50GHz");
  const std::vector<std::string> expected{"avx", "avx2", "fma", "avx512f", "avx512_vnni"};
  EXPECT_EQ(info.features, expected);
}

TEST(CpuFeatures, EmptyInputYieldsEmptyInfo) {
  const auto info = eib::parse_cpuinfo("");
  EXPECT_TRUE(info.model.empty());
  EXPECT_TRUE(info.features.empty());
}

TEST(Json, EscapesControlAndQuoteCharacters) {
  EXPECT_EQ(eib::json_escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");
  EXPECT_EQ(eib::json_escape(std::string("\x01")), "\\u0001");
}

TEST(Json, NonFiniteNumbersBecomeNull) {
  EXPECT_EQ(eib::json_number(std::nan("")), "null");
  EXPECT_EQ(eib::json_number(1.5), "1.5");
}
