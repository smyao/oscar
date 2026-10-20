// Archive #164: old/new actual SDK conversion, tails and repeat-limit guards.
// Archive #126/#145: CPU-debug is functional evidence, not asynchronous NPU proof.
#include "tikicpulib.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

extern "C" void oscar_c16_cast_pair_kernel(
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, int32_t);

namespace {
constexpr int32_t kElements = 20128, kGuard = 64;
constexpr uint16_t kSentinel = 0xa5c3;
struct Gm {
  uint8_t* p;
  size_t size;
  explicit Gm(size_t bytes)
      : p(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))), size(bytes) {
    if (!p) throw std::runtime_error("GmAlloc failed");
  }
  ~Gm() { AscendC::GmFree(p); }
  Gm(const Gm&) = delete;
};
uint16_t HalfAt(const Gm& g, int32_t i) {
  uint16_t result;
  std::memcpy(&result, g.p + i * 2, 2);
  return result;
}
uint32_t Pattern(int32_t i, int32_t variant) {
  // Positive/negative ties with odd/even low BF16 bits, signed zeros,
  // normal limits, subnormals, infinities and NaN payloads. Candidate must
  // match the original instruction for every category, including NaN bits.
  static constexpr uint32_t edges[] = {
      0x00000000, 0x80000000, 0x3f800000, 0xbf800000,
      0x3f807fff, 0x3f808000, 0x3f808001, 0x3f818000,
      0xbf807fff, 0xbf808000, 0xbf808001, 0xbf818000,
      0x007fffff, 0x00800000, 0x00800001, 0x00000001,
      0x80000001, 0x7f7f7fff, 0x7f7f8000, 0xff7f8000,
      0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
      0x7fc00000, 0x7fc10001, 0x7f800001, 0xff800001,
      0x3eaaaaab, 0x00808000, 0x80808000, 0x12345678,
      0x7f7f0000, 0x7f7f7fff, 0x7f7f8000, 0x7f7f8001,
      0xff7f0000, 0xff7f7fff, 0xff7f8000, 0xff7f8001};
  if (variant == 0) return edges[i % (sizeof(edges) / sizeof(edges[0]))];
  // Deterministic broad finite FP32 values; no floating conversion on host.
  uint32_t bits = static_cast<uint32_t>(i) * 747796405u + 2891336453u;
  bits = ((bits >> ((bits >> 28u) + 4u)) ^ bits) * 277803737u;
  bits = (bits >> 22u) ^ bits;
  return bits;
}
bool IndependentRneApplies(uint32_t bits) {
  const uint32_t magnitude = bits & 0x7fffffffu;
  // Subnormal/NaN handling is compared bit-for-bit against the original SDK
  // instruction. Independent IEEE RNE checks normal finite values and zero.
  return magnitude == 0 || (magnitude >= 0x00800000u && magnitude < 0x7f800000u);
}
uint16_t Rne(uint32_t bits, bool saturate) {
  const uint16_t rounded =
      static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  // The SDK FLOAT saturation mode is CTRL[48]==0, documented by
  // dav_c220/kernel_operator_common_impl.h:GetSaturationModeBit and
  // Set/GetSaturationFlagImpl. Apply the mode only to overflow of finite
  // inputs: preserve ordinary RNE; do not silently exempt this boundary.
  if (saturate && (bits & 0x7fffffffu) < 0x7f800000u &&
      (rounded & 0x7fffu) == 0x7f80u) {
    return static_cast<uint16_t>((rounded & 0x8000u) | 0x7f7fu);
  }
  return rounded;
}
std::string Bits(uint32_t input, uint16_t oldBits, uint16_t newBits,
                 uint16_t expectedBits) {
  std::ostringstream out;
  out << std::hex << std::setfill('0')
      << " input_bits=0x" << std::setw(8) << input
      << " old_bits=0x" << std::setw(4) << oldBits
      << " new_bits=0x" << std::setw(4) << newBits
      << " expected_bits=0x" << std::setw(4) << expectedBits;
  return out.str();
}
}  // namespace

int main() {
  try {
    const std::vector<int32_t> counts = {
        0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192,
        8193, 16319, 16320, 16321, 16383, 16384, 16385, 20000};
    Gm input(kElements * 4), original(kElements * 2), candidate(kElements * 2);
    Gm controls(32);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint64_t checked = 0, independent = 0;
    for (int32_t variant = 0; variant < 2; ++variant) {
      std::vector<uint32_t> source(kElements);
      for (int32_t i = 0; i < kElements; ++i) source[i] = Pattern(i, variant);
      std::memcpy(input.p, source.data(), input.size);
      for (const int32_t count : counts) {
        std::vector<uint16_t> before(kElements, kSentinel);
        std::memcpy(original.p, before.data(), original.size);
        std::memcpy(candidate.p, before.data(), candidate.size);
        std::memset(controls.p, 0xff, controls.size);
        ICPU_RUN_KF(oscar_c16_cast_pair_kernel, 1, input.p, original.p,
                    candidate.p, controls.p, count);
        int64_t ctrl[3];
        std::memcpy(ctrl, controls.p, sizeof(ctrl));
        if ((ctrl[0] != 0 && ctrl[0] != 1) ||
            ctrl[1] != ctrl[0] || ctrl[2] != ctrl[0])
          throw std::runtime_error("FLOAT saturation control missing or changed");
        const bool saturate = ctrl[0] == 0;
        if (std::memcmp(input.p, source.data(), input.size))
          throw std::runtime_error("source modified");
        for (int32_t i = 0; i < kElements; ++i) {
          const auto a = HalfAt(original, i), b = HalfAt(candidate, i);
          if (a != b)
            throw std::runtime_error("conversion bits differ count=" +
                std::to_string(count) + " index=" + std::to_string(i) +
                Bits(source[i], a, b, Rne(source[i], saturate)));
          if (i < kGuard || i >= kGuard + count) {
            if (a != kSentinel)
              throw std::runtime_error("inactive destination overwritten");
          } else {
            ++checked;
            if (IndependentRneApplies(source[i])) {
              if (b != Rne(source[i], saturate))
                throw std::runtime_error("independent RNE mismatch count=" +
                    std::to_string(count) + " index=" + std::to_string(i) +
                    Bits(source[i], a, b, Rne(source[i], saturate)) +
                    " float_saturation=" + std::to_string(saturate));
              ++independent;
            }
          }
        }
      }
    }
    std::cout << "{\"status\":\"passed\",\"cases\":" << counts.size() * 2
              << ",\"bitwise_elements\":" << checked
              << ",\"independent_rne_elements\":" << independent
              << ",\"source_immutable\":true,\"inactive_guards\":true,"
                 "\"float_saturation_control\":\"preserved\","
                 "\"performance\":\"not_measured\",\"NPU\":\"not_run\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "C16 cast CPU functional failure: " << error.what() << '\n';
    return 1;
  }
}
