// SPDX-License-Identifier: Apache-2.0
// Archive #21/#94/#125/#126/#140/#145/#148/#154-156; complete startup D.4:
// (1) bounded FP32 QK/PV math qualification, not full-history reconstruction;
// (2) D.4's 6.5s dequant and ~215ms store are not this probe's timing;
// (3) two arms consume the same frozen ABNZ values, preserve K128 order, and
// save every input/output/guard before paired bits and independent oracles;
// (4) no CPU wall time is a device speed result. Three finite fixtures do not
// qualify NaN/Inf/error domains, graphs, production integration or the model.
// No v1/v2 or other measured experiment is modified. No oracle fallback.
#include "tikicpulib.h"
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void oscar_fp32_sdk_kmajor_reference_kernel(
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t);
extern "C" void oscar_fp32_kmajor_matmul_kernel(
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, int64_t, int64_t);

namespace {
using Path = std::filesystem::path;
constexpr size_t kDim = 256;
constexpr size_t kGuard = 64;
constexpr double kAtol = 0.005, kRtol = 0.005;

std::vector<uint8_t> Read(const Path& path, size_t size) {
  std::vector<uint8_t> data(size);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(data.data()), size) || file.peek() != EOF)
    throw std::runtime_error("input byte count differs: " + path.string());
  return data;
}

std::string ReadText(const Path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read metadata: " + path.string());
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void Text(const Path& path, const std::string& text) {
  std::ofstream file(path, std::ios::binary);
  file << text << '\n';
  if (!file) throw std::runtime_error("cannot save " + path.string());
}

size_t SkipSpace(const std::string& text, size_t pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
  return pos;
}

// Deliberately narrow parser for this fixed fixture schema. Require one exact
// key and a complete literal, rather than finding "true" anywhere in JSON.
size_t FieldStart(const std::string& text, const std::string& key) {
  const std::string quoted = "\"" + key + "\"";
  const size_t pos = text.find(quoted);
  if (pos == std::string::npos || text.find(quoted, pos + quoted.size()) != std::string::npos)
    throw std::runtime_error("missing or duplicate metadata key: " + key);
  size_t next = SkipSpace(text, pos + quoted.size());
  if (next >= text.size() || text[next] != ':')
    throw std::runtime_error("invalid metadata field: " + key);
  return SkipSpace(text, next + 1);
}

void FieldEnd(const std::string& text, size_t pos) {
  pos = SkipSpace(text, pos);
  if (pos >= text.size() || (text[pos] != ',' && text[pos] != '}'))
    throw std::runtime_error("invalid metadata literal terminator");
}

bool BoolField(const std::string& text, const std::string& key) {
  const size_t pos = FieldStart(text, key);
  if (text.compare(pos, 4, "true") == 0) { FieldEnd(text, pos + 4); return true; }
  if (text.compare(pos, 5, "false") == 0) { FieldEnd(text, pos + 5); return false; }
  throw std::runtime_error("boolean metadata literal required: " + key);
}

std::string StringField(const std::string& text, const std::string& key) {
  const size_t pos = FieldStart(text, key);
  if (pos >= text.size() || text[pos] != '"') throw std::runtime_error("string field required");
  const size_t end = text.find('"', pos + 1);
  if (end == std::string::npos) throw std::runtime_error("unterminated string field");
  const auto value = text.substr(pos + 1, end - pos - 1);
  if (value.find('\\') != std::string::npos) throw std::runtime_error("escaped case name is not allowed");
  FieldEnd(text, end + 1);
  return value;
}

double NumberField(const std::string& text, const std::string& key) {
  const size_t pos = FieldStart(text, key);
  size_t used = 0;
  const double value = std::stod(text.substr(pos), &used);
  FieldEnd(text, pos + used);
  return value;
}

void FrozenTolerance(const std::string& text) {
  const size_t start = FieldStart(text, "independent_oracle_tolerance");
  if (start >= text.size() || text[start] != '{') throw std::runtime_error("oracle tolerance object required");
  const size_t end = text.find('}', start + 1);
  if (end == std::string::npos) throw std::runtime_error("unterminated oracle tolerance object");
  const auto object = text.substr(start, end - start + 1);
  if (NumberField(object, "atol") != kAtol || NumberField(object, "rtol") != kRtol)
    throw std::runtime_error("frozen fused-attention tolerance differs");
}

uint32_t Word(const uint8_t* data, size_t index) {
  uint32_t word; std::memcpy(&word, data + index * 4, 4); return word;
}
float Float(const uint8_t* data, size_t index) {
  float value; std::memcpy(&value, data + index * 4, 4); return value;
}
size_t Differences(const std::vector<uint8_t>& left, const std::vector<uint8_t>& right,
                   size_t begin, size_t end) {
  size_t differences = 0;
  for (size_t i = begin; i < end; ++i) differences += left[i] != right[i];
  return differences;
}
const char* Boolean(bool value) { return value ? "true" : "false"; }

struct Gm {
  uint8_t* base; uint8_t* ptr; size_t size; std::vector<uint8_t> initial;
  explicit Gm(size_t bytes) : size(bytes), initial(bytes, 0x85) {
    base = static_cast<uint8_t*>(AscendC::GmAlloc(size + 2 * kGuard));
    if (!base) throw std::runtime_error("GmAlloc failed");
    ptr = base + kGuard;
    std::memset(base, 0xa5, size + 2 * kGuard);
    std::memcpy(ptr, initial.data(), size);
  }
  ~Gm() { AscendC::GmFree(base); }
  Gm(const Gm&) = delete;
  void Load(const std::vector<uint8_t>& data) {
    if (data.size() != size) throw std::runtime_error("GM input size mismatch");
    initial = data; std::memcpy(ptr, data.data(), size);
  }
  bool Guards() const {
    for (size_t i = 0; i < kGuard; ++i)
      if (base[i] != 0xa5 || ptr[size + i] != 0xa5) return false;
    return true;
  }
  bool Unchanged() const { return !std::memcmp(ptr, initial.data(), size); }
  void Save(const Path& output, const std::string& name) const {
    std::ofstream plain(output / (name + ".bin"), std::ios::binary);
    std::ofstream guarded(output / (name + ".guarded.bin"), std::ios::binary);
    plain.write(reinterpret_cast<const char*>(ptr), size);
    guarded.write(reinterpret_cast<const char*>(base), size + 2 * kGuard);
    if (!plain || !guarded) throw std::runtime_error("cannot save " + name);
  }
};

struct ArmResult {
  std::vector<uint8_t> output, status;
  bool guards, inputs;
};

bool Completion(const std::vector<uint8_t>& bytes, int arm, int64_t rows,
                int64_t padded, int64_t layout) {
  const uint32_t fields[11] = {1, static_cast<uint32_t>(arm), static_cast<uint32_t>(rows),
      static_cast<uint32_t>(layout), 4, 64, 64, 128, 128, 256, 256};
  for (size_t i = 0; i < 11; ++i) if (Word(bytes.data(), i) != fields[i]) return false;
  return Word(bytes.data(), 15) == 0 && Word(bytes.data(), 25) == 8 &&
         Word(bytes.data(), 27) == static_cast<uint32_t>(padded) &&
         Word(bytes.data(), 28) == static_cast<uint32_t>(padded) &&
         Word(bytes.data(), 29) == static_cast<uint32_t>(rows);
}
}  // namespace

int main(int argc, char** argv) { try {
  if (argc != 3) throw std::runtime_error("usage: oscar_fp32_kmajor_matmul_cpu FIXTURE NEW_OUTPUT");
  const Path fixture = argv[1], output = argv[2];
  int64_t rows, padded, layout;
  std::ifstream shape(fixture / "shape.txt");
  shape >> rows >> padded >> layout;
  std::string extra;
  if (!shape || shape >> extra ||
      !((rows == 24 && padded == 32 && layout == 0) ||
        (rows == 126 && padded == 128 && layout == 1)))
    throw std::runtime_error("undeclared shape");
  const auto metadata = ReadText(fixture / "fixture.json");
  const std::string name = StringField(metadata, "case");
  const bool exact = BoolField(metadata, "exact_integer_oracle");
  if (!((name == "qk_m24" && exact && rows == 24 && layout == 0) ||
        (name == "pv_m126" && exact && rows == 126 && layout == 1) ||
        (name == "cancel_qk_m24" && !exact && rows == 24 && layout == 0)))
    throw std::runtime_error("case/oracle/shape identity differs");
  FrozenTolerance(metadata);
  const size_t elements = static_cast<size_t>(padded) * kDim, live = static_cast<size_t>(rows) * kDim;
  auto inputA = Read(fixture / "a_nz.bin", elements * 4);
  auto inputB = Read(fixture / "b_nz.bin", kDim * kDim * 4);
  auto expected = Read(fixture / "expected_output.bin", elements * 4);
  for (size_t r = rows; r < static_cast<size_t>(padded); ++r)
    for (size_t c = 0; c < kDim; ++c)
      if (Word(inputA.data(), (c / 8) * padded * 8 + r * 8 + c % 8) != 0)
        throw std::runtime_error("A NZ input padding is not positive zero");
  for (size_t i = live; i < elements; ++i)
    if (Word(expected.data(), i) != 0x85858585) throw std::runtime_error("oracle padding poison differs");
  if (std::filesystem::exists(output)) throw std::runtime_error("refuse existing output");
  std::filesystem::create_directories(output);
  Text(output / "fixture.json", metadata);
  AscendC::SetKernelMode(KernelMode::AIC_MODE);
  std::vector<ArmResult> results;
  const std::string names[2] = {"sdk", "kmajor"};
  for (int arm = 0; arm < 2; ++arm) {
    const Path dir = output / names[arm]; std::filesystem::create_directory(dir);
    Gm a(elements * 4), b(kDim * kDim * 4), c(elements * 4), status(128);
    a.Load(inputA); b.Load(inputB);
    std::cout << "KMAJOR_CPU_ARM_START " << names[arm] << " case=" << name << std::endl;
    if (arm == 0) {
      ICPU_RUN_KF(oscar_fp32_sdk_kmajor_reference_kernel, 1, a.ptr, b.ptr, c.ptr, status.ptr, rows, layout);
    } else {
      ICPU_RUN_KF(oscar_fp32_kmajor_matmul_kernel, 1, a.ptr, b.ptr, c.ptr, status.ptr, rows, layout);
    }
    a.Save(dir, "a_after"); b.Save(dir, "b_after"); c.Save(dir, "c"); status.Save(dir, "status");
    // Snapshot after complete saved evidence. No numerical/status gate can
    // skip the second arm or hide a failing SDK reference oracle.
    results.push_back({std::vector<uint8_t>(c.ptr, c.ptr + c.size),
                       std::vector<uint8_t>(status.ptr, status.ptr + status.size),
                       a.Guards() && b.Guards() && c.Guards() && status.Guards(),
                       a.Unchanged() && b.Unchanged()});
  }

  // Paired bytes, INCLUDING padding, are examined before either oracle.
  const size_t pair = Differences(results[0].output, results[1].output, 0, elements * 4);
  std::string first = "["; size_t shown = 0;
  for (size_t i = 0; i < elements && shown < 8; ++i) {
    const uint32_t left = Word(results[0].output.data(), i), right = Word(results[1].output.data(), i);
    if (left == right) continue;
    if (shown++) first += ",";
    first += "{\"element\":" + std::to_string(i) + ",\"sdk_bits\":" + std::to_string(left) +
             ",\"kmajor_bits\":" + std::to_string(right) + "}";
  }
  first += "]";
  bool passed = pair == 0; std::string arms = "[";
  for (int arm = 0; arm < 2; ++arm) {
    const auto& result = results[arm]; size_t oracleBad = 0, unexpectedNonfinite = 0;
    for (size_t i = 0; i < live; ++i) {
      const float actual = Float(result.output.data(), i), wanted = Float(expected.data(), i);
      const bool finite = std::isfinite(actual) && std::isfinite(wanted);
      unexpectedNonfinite += !finite;
      if (exact) oracleBad += !finite || Word(result.output.data(), i) != Word(expected.data(), i);
      else oracleBad += !finite || std::fabs(static_cast<double>(actual) - wanted) >
                                     kAtol + kRtol * std::fabs(static_cast<double>(wanted));
    }
    const size_t padding = Differences(result.output, expected, live * 4, elements * 4);
    const bool complete = Completion(result.status, arm, rows, padded, layout);
    const bool ok = result.guards && result.inputs && complete && oracleBad == 0 && padding == 0;
    passed = passed && ok;
    std::string fields = "[";
    for (size_t i = 0; i < 32; ++i) { if (i) fields += ","; fields += std::to_string(Word(result.status.data(), i)); }
    fields += "]";
    const std::string row = "{\"arm\":\"" + names[arm] + "\",\"status\":\"" + (ok ? "passed" : "failed") +
      "\",\"exact_integer_oracle\":" + Boolean(exact) + ",\"oracle_live_bad_elements\":" + std::to_string(oracleBad) +
      ",\"unexpected_nonfinite_elements\":" + std::to_string(unexpectedNonfinite) +
      ",\"padding_byte_changes\":" + std::to_string(padding) + ",\"guards\":" + Boolean(result.guards) +
      ",\"inputs_unchanged\":" + Boolean(result.inputs) + ",\"completion\":" + Boolean(complete) +
      ",\"status_fields\":" + fields + "}";
    Text(output / names[arm] / "report.json", row);
    if (arm) arms += ","; arms += row;
    std::cout << "KMAJOR_CPU_ARM " << row << std::endl;
  }
  arms += "]";
  const std::string report = "{\"status\":\"" + std::string(passed ? "passed" : "failed") +
      "\",\"case\":\"" + name + "\",\"backend\":\"official_cpu_debug\",\"paired_output_byte_differences\":" +
      std::to_string(pair) + ",\"paired_first_differences\":" + first + ",\"arms\":" + arms +
      ",\"NPU\":\"not_run\",\"graph\":\"not_run\",\"performance\":\"not_measured\"}";
  Text(output / "report.json", report);
  std::cout << "KMAJOR_CPU_RESULT " << report << std::endl;
  return passed ? 0 : 2;
} catch (const std::exception& error) {
  std::cerr << "KMAJOR_CPU_ERROR " << error.what() << std::endl;
  return 2;
} }
