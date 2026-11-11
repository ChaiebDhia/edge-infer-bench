// edge_bench: latency / throughput / memory benchmark for ONNX models on CPU.
//
// Usage:
//   edge_bench --model m.onnx --label fp32 --threads 4 --iters 100 --warmup 10
//              [--batch 1] [--shape 1,3,300,300] [--seed 42] [--out result.json]
//
// One JSON document is written to --out (or stdout). It records the machine
// (architecture, CPU model, ML-relevant instruction extensions) next to the
// numbers, because a latency figure without its hardware context is meaningless.

#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_features.hpp"
#include "json.hpp"
#include "stats.hpp"

namespace {

struct Args {
  std::string model;
  std::string label = "model";
  std::string out;
  std::string shape;  // optional override, e.g. "1,3,300,300"
  int threads = 1;
  int iters = 100;
  int warmup = 10;
  int batch = 1;
  unsigned seed = 42;
};

[[noreturn]] void usage_and_exit(const std::string& msg) {
  std::cerr << "error: " << msg << "\n"
            << "usage: edge_bench --model FILE [--label NAME] [--threads N] [--iters N]\n"
            << "                  [--warmup N] [--batch N] [--shape D0,D1,...] [--seed N] [--out FILE]\n";
  std::exit(2);
}

Args parse_args(int argc, char** argv) {
  std::map<std::string, std::string> kv;
  for (int i = 1; i < argc; i += 2) {
    const std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) usage_and_exit("malformed argument near '" + key + "'");
    kv[key.substr(2)] = argv[i + 1];
  }
  Args a;
  auto take = [&](const char* name, auto& dst, auto convert) {
    auto it = kv.find(name);
    if (it != kv.end()) {
      dst = convert(it->second);
      kv.erase(it);
    }
  };
  auto to_str = [](const std::string& s) { return s; };
  auto to_int = [](const std::string& s) {
    try {
      return std::stoi(s);
    } catch (...) {
      usage_and_exit("not an integer: " + s);
    }
  };
  take("model", a.model, to_str);
  take("label", a.label, to_str);
  take("out", a.out, to_str);
  take("shape", a.shape, to_str);
  take("threads", a.threads, to_int);
  take("iters", a.iters, to_int);
  take("warmup", a.warmup, to_int);
  take("batch", a.batch, to_int);
  int seed = static_cast<int>(a.seed);
  take("seed", seed, to_int);
  a.seed = static_cast<unsigned>(seed);
  if (!kv.empty()) usage_and_exit("unknown option --" + kv.begin()->first);
  if (a.model.empty()) usage_and_exit("--model is required");
  if (a.threads < 1 || a.iters < 1 || a.warmup < 0 || a.batch < 1) usage_and_exit("numeric option out of range");
  return a;
}

std::vector<int64_t> parse_shape(const std::string& s) {
  std::vector<int64_t> dims;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) dims.push_back(std::stoll(tok));
  return dims;
}

double peak_rss_mb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return static_cast<double>(ru.ru_maxrss) / 1024.0;  // Linux reports KiB
}

std::string file_size_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  return f ? std::to_string(static_cast<long long>(f.tellg())) : "0";
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge_bench");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(args.threads);
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, args.model.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;

    if (session.GetInputCount() != 1) usage_and_exit("only single-input models are supported");
    const auto in_name = session.GetInputNameAllocated(0, alloc);
    // Keep the TypeInfo alive: the shape info below is a view into it.
    const Ort::TypeInfo in_type = session.GetInputTypeInfo(0);
    const auto in_info = in_type.GetTensorTypeAndShapeInfo();
    if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      usage_and_exit("only float32 model inputs are supported");
    }

    std::vector<int64_t> shape = args.shape.empty() ? in_info.GetShape() : parse_shape(args.shape);
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0) {
        if (i == 0)
          shape[i] = args.batch;
        else
          usage_and_exit("model has a dynamic non-batch dimension; pass --shape");
      }
    }
    std::size_t numel = 1;
    for (auto d : shape) numel *= static_cast<std::size_t>(d);

    std::vector<float> input(numel);
    std::mt19937 rng(args.seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : input) x = dist(rng);

    const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());

    const auto out_name = session.GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    auto run_once = [&]() { return session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1); };

    for (int i = 0; i < args.warmup; ++i) run_once();

    std::vector<double> lat_ms;
    lat_ms.reserve(static_cast<std::size_t>(args.iters));
    std::vector<Ort::Value> last;
    const auto wall0 = std::chrono::steady_clock::now();
    for (int i = 0; i < args.iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      last = run_once();
      const auto t1 = std::chrono::steady_clock::now();
      lat_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    const eib::Summary s = eib::summarize(lat_ms);

    // Output sanity data: top-1 of the first batch element and a checksum, so a
    // broken quantized model (NaNs, all-zero logits) is caught, not just a fast one.
    std::string top1 = "null", checksum = "null", classes = "null", finite = "true";
    if (!last.empty() && last[0].IsTensor()) {
      const auto oi = last[0].GetTensorTypeAndShapeInfo();
      if (oi.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && oi.GetElementCount() > 0) {
        const float* p = last[0].GetTensorData<float>();
        const auto shp = oi.GetShape();
        const std::size_t per = static_cast<std::size_t>(shp.empty() ? 1 : shp.back());
        std::size_t best = 0;
        double sum = 0.0;
        bool all_finite = true;
        for (std::size_t i = 0; i < oi.GetElementCount(); ++i) {
          if (!std::isfinite(p[i])) all_finite = false;
          sum += static_cast<double>(p[i]);
          if (i < per && p[i] > p[best]) best = i;
        }
        top1 = std::to_string(best);
        checksum = eib::json_number(sum);
        classes = std::to_string(per);
        finite = all_finite ? "true" : "false";
      }
    }

    const eib::CpuInfo cpu = eib::read_cpuinfo();
    std::ostringstream feats;
    for (std::size_t i = 0; i < cpu.features.size(); ++i) feats << (i ? "," : "") << eib::json_string(cpu.features[i]);

#if defined(__aarch64__)
    const char* arch = "aarch64";
#elif defined(__x86_64__)
    const char* arch = "x86_64";
#else
    const char* arch = "other";
#endif

    std::ostringstream js;
    js << "{\n"
       << "  \"label\": " << eib::json_string(args.label) << ",\n"
       << "  \"model\": " << eib::json_string(args.model) << ",\n"
       << "  \"model_bytes\": " << file_size_bytes(args.model) << ",\n"
       << "  \"threads\": " << args.threads << ",\n"
       << "  \"batch\": " << shape[0] << ",\n"
       << "  \"iters\": " << args.iters << ",\n"
       << "  \"warmup\": " << args.warmup << ",\n"
       << "  \"latency_ms\": {\"mean\": " << eib::json_number(s.mean) << ", \"stddev\": " << eib::json_number(s.stddev)
       << ", \"min\": " << eib::json_number(s.min) << ", \"p50\": " << eib::json_number(s.p50)
       << ", \"p95\": " << eib::json_number(s.p95) << ", \"p99\": " << eib::json_number(s.p99)
       << ", \"max\": " << eib::json_number(s.max) << "},\n"
       << "  \"throughput_ips\": "
       << eib::json_number(static_cast<double>(args.iters) * static_cast<double>(shape[0]) / wall_s) << ",\n"
       << "  \"peak_rss_mb\": " << eib::json_number(peak_rss_mb()) << ",\n"
       << "  \"output\": {\"top1\": " << top1 << ", \"checksum\": " << checksum << ", \"classes\": " << classes
       << ", \"all_finite\": " << finite << "},\n"
       << "  \"system\": {\"arch\": " << eib::json_string(arch) << ", \"cpu_model\": " << eib::json_string(cpu.model)
       << ", \"features\": [" << feats.str() << "], \"ort_version\": " << eib::json_string(Ort::GetVersionString())
       << ", \"compiler\": " << eib::json_string(__VERSION__) << "}\n"
       << "}\n";

    if (args.out.empty()) {
      std::cout << js.str();
    } else {
      std::ofstream f(args.out);
      if (!f) usage_and_exit("cannot write --out file");
      f << js.str();
    }
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "onnxruntime error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
