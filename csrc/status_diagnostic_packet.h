// Archive #157: host-only formatting of the bounded device diagnostic packet.
#pragma once
#include <cstdint>
#include <sstream>
#include <string>

namespace oscar_diagnostic {
constexpr int64_t kGroups = 4;
constexpr int64_t kWidth = 32;
inline std::string ErrorLines(const int64_t* packet, const std::string& context,
                              int pid, int device, uint64_t invocation) {
    constexpr const char* groups[] = {"attention", "rotate", "merge", "store"};
    constexpr const char* fields[] = {"index", "code", "nonzero_count", "size", "token", "head",
        "lane", "slot", "position", "task_index", "query_begin", "query_end", "seq_len",
        "virtual_block", "expected_slot", "physical_page"};
    std::ostringstream out;
    for (int g = 0; g < kGroups; ++g) {
        const auto* row = packet + g * kWidth;
        if (row[2] == 0) continue;
        out << "OSCAR_FIRST_ERROR {\"pid\":" << pid << ",\"device\":" << device
            << ",\"callsite_invocation\":" << invocation << ",\"context\":" << context
            << ",\"group\":\"" << groups[g] << "\"";
        for (int i = 0; i < 16; ++i) out << ",\"" << fields[i] << "\":" << row[i];
        out << ",\"task\":[";
        for (int i = 16; i < kWidth; ++i) out << (i == 16 ? "" : ",") << row[i];
        out << "]}\n";
    }
    return out.str();
}
}
