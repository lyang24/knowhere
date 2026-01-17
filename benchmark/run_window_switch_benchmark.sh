#!/bin/bash
# Window Switch Benchmark Runner
# Run this on an AWS instance (recommended: c6i.xlarge or larger)
#
# Prerequisites:
#   1. Install dependencies: ./scripts/install_deps.sh
#   2. Install Conan packages (see below)
#
# Quick start on Ubuntu 20.04/22.04:
#   sudo apt update
#   sudo apt install -y g++ gcc make cmake python3-pip libboost-all-dev
#   pip3 install conan==1.61.0
#   conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local
#   cd knowhere && conan install . --build=missing -o with_ut=False
#   ./benchmark/run_window_switch_benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KNOWHERE_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${KNOWHERE_ROOT}/build"

echo "=============================================="
echo "  Window Switch Benchmark"
echo "=============================================="

# Print system info
echo ""
echo "System Information:"
echo "  Hostname: $(hostname)"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs 2>/dev/null || echo 'Unknown')"
echo "  Cores: $(nproc)"
echo ""

# Print cache info from system
echo "System Cache Info (from OS):"
if [ -f /sys/devices/system/cpu/cpu0/cache/index0/size ]; then
    echo "  L1d: $(cat /sys/devices/system/cpu/cpu0/cache/index0/size 2>/dev/null || echo 'N/A')"
    echo "  L2:  $(cat /sys/devices/system/cpu/cpu0/cache/index2/size 2>/dev/null || echo 'N/A')"
    echo "  L3:  $(cat /sys/devices/system/cpu/cpu0/cache/index3/size 2>/dev/null || echo 'N/A')"
else
    lscpu | grep -i cache || true
fi
echo ""

# Check if benchmark exists
if [ ! -f "${BUILD_DIR}/benchmark_window_switch" ]; then
    echo "Benchmark not found at ${BUILD_DIR}/benchmark_window_switch"
    echo ""
    echo "Please build knowhere first:"
    echo "  1. Install dependencies:"
    echo "     ./scripts/install_deps.sh"
    echo ""
    echo "  2. Install Conan packages:"
    echo "     conan install . --build=missing -o with_ut=False"
    echo ""
    echo "  3. Build:"
    echo "     mkdir -p build && cd build"
    echo "     cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "     make -j\$(nproc) benchmark_window_switch"
    echo ""
    exit 1
fi

echo "=============================================="
echo "  Running Benchmark"
echo "=============================================="
echo ""

# Run the benchmark
cd "${BUILD_DIR}"
./benchmark_window_switch

echo ""
echo "=============================================="
echo "  Benchmark Complete"
echo "=============================================="
