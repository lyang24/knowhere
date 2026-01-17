// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_SIMD_CACHE_INFO_H
#define KNOWHERE_SIMD_CACHE_INFO_H

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <fstream>
#include <string>
#endif

namespace knowhere {

// Detects CPU cache sizes at runtime for cache-aware optimizations.
// Used by Window Switch and other cache-sensitive algorithms.
class CacheInfo {
 public:
    static CacheInfo&
    GetInstance() {
        static CacheInfo instance;
        return instance;
    }

    // Returns L3 cache size in bytes, or 0 if detection failed
    size_t
    L3CacheSize() const {
        return l3_cache_size_;
    }

    // Returns L2 cache size in bytes, or 0 if detection failed
    size_t
    L2CacheSize() const {
        return l2_cache_size_;
    }

    // Returns L1 data cache size in bytes, or 0 if detection failed
    size_t
    L1DataCacheSize() const {
        return l1d_cache_size_;
    }

    // Returns recommended window size for cache-aware algorithms.
    // This is the number of float elements that fit in ~50% of L3 cache,
    // leaving room for posting lists and other working data.
    // Falls back to 100,000 if cache detection fails.
    size_t
    RecommendedWindowSize() const {
        if (l3_cache_size_ > 0) {
            // Use 50% of L3 cache for the scores array
            // Each score is 4 bytes (float)
            return (l3_cache_size_ / 2) / sizeof(float);
        }
        // Fallback: assume 8MB L3 cache (common on modern CPUs)
        // 8MB * 50% / 4 bytes = 1,000,000 elements
        // But we cap at 100,000 to be conservative
        return 100000;
    }

 private:
    CacheInfo() : l1d_cache_size_(0), l2_cache_size_(0), l3_cache_size_(0) {
        DetectCacheSizes();
    }

    void
    DetectCacheSizes() {
#if defined(__x86_64__) || defined(_M_X64)
        DetectCacheSizesX86();
#elif defined(__APPLE__)
        DetectCacheSizesMacOS();
#elif defined(__linux__)
        DetectCacheSizesLinux();
#endif
    }

#if defined(__x86_64__) || defined(_M_X64)
    void
    DetectCacheSizesX86() {
        // Use CPUID leaf 0x04 to enumerate cache parameters
        // This works on Intel and AMD processors
        uint32_t eax, ebx, ecx, edx;

        for (int cache_index = 0; cache_index < 16; ++cache_index) {
            __cpuid_count(0x04, cache_index, eax, ebx, ecx, edx);

            // Check cache type (bits 4:0 of EAX)
            // 0 = null (no more caches), 1 = data, 2 = instruction, 3 = unified
            uint32_t cache_type = eax & 0x1F;
            if (cache_type == 0) {
                break;  // No more caches
            }

            // Skip instruction caches
            if (cache_type == 2) {
                continue;
            }

            // Cache level (bits 7:5 of EAX)
            uint32_t cache_level = (eax >> 5) & 0x7;

            // Calculate cache size:
            // Size = (Ways + 1) * (Partitions + 1) * (Line Size + 1) * (Sets + 1)
            uint32_t ways = ((ebx >> 22) & 0x3FF) + 1;
            uint32_t partitions = ((ebx >> 12) & 0x3FF) + 1;
            uint32_t line_size = (ebx & 0xFFF) + 1;
            uint32_t sets = ecx + 1;

            size_t cache_size = static_cast<size_t>(ways) * partitions * line_size * sets;

            switch (cache_level) {
                case 1:
                    if (cache_type == 1 || cache_type == 3) {  // Data or unified
                        l1d_cache_size_ = cache_size;
                    }
                    break;
                case 2:
                    l2_cache_size_ = cache_size;
                    break;
                case 3:
                    l3_cache_size_ = cache_size;
                    break;
            }
        }
    }
#endif

#if defined(__APPLE__)
    void
    DetectCacheSizesMacOS() {
        size_t size;
        size_t len = sizeof(size);

        // macOS uses sysctl for cache info
        if (sysctlbyname("hw.l1dcachesize", &size, &len, nullptr, 0) == 0) {
            l1d_cache_size_ = size;
        }
        if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0) {
            l2_cache_size_ = size;
        }
        if (sysctlbyname("hw.l3cachesize", &size, &len, nullptr, 0) == 0) {
            l3_cache_size_ = size;
        }
    }
#endif

#if defined(__linux__) && !defined(__x86_64__) && !defined(_M_X64)
    void
    DetectCacheSizesLinux() {
        // On Linux ARM, read from sysfs
        // /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/
        for (int index = 0; index < 4; ++index) {
            std::string base_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";

            // Read cache level
            std::ifstream level_file(base_path + "level");
            int level = 0;
            if (level_file >> level) {
                // Read cache type
                std::ifstream type_file(base_path + "type");
                std::string type;
                if (type_file >> type) {
                    // Skip instruction caches
                    if (type == "Instruction") {
                        continue;
                    }
                }

                // Read cache size (e.g., "32K", "256K", "8192K")
                std::ifstream size_file(base_path + "size");
                std::string size_str;
                if (size_file >> size_str) {
                    size_t size = 0;
                    size_t multiplier = 1;

                    // Parse size string
                    if (!size_str.empty()) {
                        char suffix = size_str.back();
                        if (suffix == 'K' || suffix == 'k') {
                            multiplier = 1024;
                            size_str.pop_back();
                        } else if (suffix == 'M' || suffix == 'm') {
                            multiplier = 1024 * 1024;
                            size_str.pop_back();
                        }
                        size = std::stoull(size_str) * multiplier;
                    }

                    switch (level) {
                        case 1:
                            l1d_cache_size_ = size;
                            break;
                        case 2:
                            l2_cache_size_ = size;
                            break;
                        case 3:
                            l3_cache_size_ = size;
                            break;
                    }
                }
            }
        }
    }
#endif

    size_t l1d_cache_size_;
    size_t l2_cache_size_;
    size_t l3_cache_size_;
};

}  // namespace knowhere

#endif  // KNOWHERE_SIMD_CACHE_INFO_H
