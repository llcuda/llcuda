#!/bin/bash
################################################################################
# Unified CUDA 12 Build Script for Multiple GPU Targets
# Supports: GeForce 940M (CC 5.0) and Tesla T4 (CC 7.5)
#
# Usage:
#   ./build_cuda12_unified.sh 940m          # Build for GeForce 940M
#   ./build_cuda12_unified.sh t4            # Build for Tesla T4
#   ./build_cuda12_unified.sh both          # Build for both GPUs
#   ./build_cuda12_unified.sh auto          # Auto-detect GPU and build
################################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# GPU Profiles
declare -A GPU_PROFILES

GPU_PROFILES[940m_name]="GeForce 940M"
GPU_PROFILES[940m_cc]="50"
GPU_PROFILES[940m_arch]="Maxwell"
GPU_PROFILES[940m_vram]="1GB"
GPU_PROFILES[940m_fa]="OFF"
GPU_PROFILES[940m_cublas]="ON"
GPU_PROFILES[940m_mmq]="OFF"

GPU_PROFILES[t4_name]="Tesla T4"
GPU_PROFILES[t4_cc]="75"
GPU_PROFILES[t4_arch]="Turing"
GPU_PROFILES[t4_vram]="15GB"
GPU_PROFILES[t4_fa]="ON"
GPU_PROFILES[t4_cublas]="OFF"
GPU_PROFILES[t4_mmq]="OFF"

# Configuration
PROJECT_DIR="/media/waqasm86/External1/Project-Nvidia"
LLAMA_CPP_DIR="${PROJECT_DIR}/llama.cpp"

# Detect CUDA
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
for cuda_dir in /usr/local/cuda-12.* /usr/local/cuda; do
    if [ -d "$cuda_dir" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
        CUDA_HOME="$cuda_dir"
        break
    fi
done
CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

################################################################################
# Functions
################################################################################

show_help() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}Unified CUDA 12 Build Script for llama.cpp${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo -e "  $0 [target]"
    echo ""
    echo -e "${GREEN}Targets:${NC}"
    echo -e "  ${CYAN}940m${NC}     - Build for NVIDIA GeForce 940M (CC 5.0)"
    echo -e "  ${CYAN}t4${NC}       - Build for NVIDIA Tesla T4 (CC 7.5)"
    echo -e "  ${CYAN}both${NC}     - Build for both GPUs"
    echo -e "  ${CYAN}auto${NC}     - Auto-detect GPU and build"
    echo -e "  ${CYAN}help${NC}     - Show this help message"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo -e "  $0 940m"
    echo -e "  $0 t4"
    echo -e "  $0 auto"
    echo ""
    exit 0
}

detect_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo ""
        return 1
    fi

    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

    if [[ "$gpu_name" == *"940M"* ]]; then
        echo "940m"
    elif [[ "$gpu_name" == *"T4"* ]]; then
        echo "t4"
    else
        # Try to get compute capability
        local cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [ ! -z "$cc" ]; then
            if [ "$cc" -le 52 ]; then
                echo "940m"
            elif [ "$cc" -ge 70 ]; then
                echo "t4"
            fi
        fi
    fi
}

print_gpu_profile() {
    local profile=$1
    echo -e "${CYAN}GPU Profile: ${GPU_PROFILES[${profile}_name]}${NC}"
    echo -e "  Architecture:       ${GPU_PROFILES[${profile}_arch]}"
    echo -e "  Compute Capability: ${GPU_PROFILES[${profile}_cc]}"
    echo -e "  VRAM:               ${GPU_PROFILES[${profile}_vram]}"
    echo -e "  FlashAttention:     ${GPU_PROFILES[${profile}_fa]}"
    echo -e "  Force cuBLAS:       ${GPU_PROFILES[${profile}_cublas]}"
}

build_for_gpu() {
    local profile=$1
    local cc=${GPU_PROFILES[${profile}_cc]}
    local name=${GPU_PROFILES[${profile}_name]}
    local fa=${GPU_PROFILES[${profile}_fa]}
    local cublas=${GPU_PROFILES[${profile}_cublas]}
    local mmq=${GPU_PROFILES[${profile}_mmq]}

    local build_dir="${LLAMA_CPP_DIR}/build_cuda12_${profile}"
    local install_dir="${PROJECT_DIR}/llcuda/llcuda/binaries/cuda12_${profile}"
    local lib_dir="${PROJECT_DIR}/llcuda/llcuda/lib"

    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}Building for ${name} (Profile: ${profile})${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo ""

    print_gpu_profile "$profile"
    echo ""
    echo -e "${YELLOW}Build Configuration:${NC}"
    echo -e "  Source:      ${LLAMA_CPP_DIR}"
    echo -e "  Build Dir:   ${build_dir}"
    echo -e "  Install Dir: ${install_dir}"
    echo -e "  CUDA Home:   ${CUDA_HOME}"
    echo ""

    # Clean and create directories
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    mkdir -p "${install_dir}"
    mkdir -p "${lib_dir}"

    # CMake configuration
    echo -e "${YELLOW}Configuring with CMake...${NC}"
    cd "${LLAMA_CPP_DIR}"

    cmake -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${cc}" \
        -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA_FORCE_MMQ=${mmq} \
        -DGGML_CUDA_FORCE_CUBLAS=${cublas} \
        -DGGML_CUDA_FA=${fa} \
        -DGGML_CUDA_FA_ALL_QUANTS=${fa} \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_CURL=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ CMake configuration failed${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ CMake configuration completed${NC}"
    echo ""

    # Build
    echo -e "${YELLOW}Building (this may take 10-30 minutes)...${NC}"
    local num_threads=$(nproc)
    cmake --build "${build_dir}" --config Release -j${num_threads}

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Build failed${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Build completed${NC}"
    echo ""

    # Install
    echo -e "${YELLOW}Installing binaries and libraries...${NC}"

    # Copy binaries
    for binary in llama-server llama-cli llama-quantize llama-embedding llama-bench; do
        if [ -f "${build_dir}/bin/${binary}" ]; then
            cp "${build_dir}/bin/${binary}" "${install_dir}/"
            chmod +x "${install_dir}/${binary}"
            echo -e "  ✓ ${binary}"
        fi
    done

    # Copy libraries
    cp "${build_dir}"/bin/libllama.so* "${lib_dir}/" 2>/dev/null || true
    cp "${build_dir}"/bin/libggml*.so* "${lib_dir}/" 2>/dev/null || true
    cp "${build_dir}"/ggml/src/libggml*.so* "${lib_dir}/" 2>/dev/null || true

    echo -e "${GREEN}✓ Installation completed${NC}"
    echo ""

    # Verification
    echo -e "${YELLOW}Verification:${NC}"
    echo -e "  Binaries:"
    ls -lh "${install_dir}/" | grep -v "^total" | awk '{print "    " $9 " (" $5 ")"}'
    echo -e "  Libraries:"
    ls -lh "${lib_dir}/" | grep -v "^total" | head -5 | awk '{print "    " $9 " (" $5 ")"}'
    echo ""

    echo -e "${GREEN}✓ Build for ${name} completed successfully${NC}"
    echo ""
}

################################################################################
# Main Script
################################################################################

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Unified CUDA 12 Build Script${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Parse arguments
TARGET="${1:-help}"

case "$TARGET" in
    help|--help|-h)
        show_help
        ;;
    auto)
        echo -e "${YELLOW}Auto-detecting GPU...${NC}"
        DETECTED=$(detect_gpu)
        if [ -z "$DETECTED" ]; then
            echo -e "${RED}✗ Could not detect GPU${NC}"
            echo -e "${YELLOW}Please specify target manually: 940m or t4${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Detected: ${GPU_PROFILES[${DETECTED}_name]}${NC}"
        echo ""
        TARGET="$DETECTED"
        ;;
esac

# Verify CUDA installation
echo -e "${YELLOW}Verifying CUDA installation...${NC}"
if [ ! -f "${CUDA_COMPILER}" ]; then
    echo -e "${RED}✗ CUDA compiler not found at ${CUDA_COMPILER}${NC}"
    echo -e "${RED}Please install CUDA 12.x or set CUDA_HOME environment variable${NC}"
    exit 1
fi

${CUDA_COMPILER} --version | head -1
echo -e "${GREEN}✓ CUDA found at ${CUDA_HOME}${NC}"
echo ""

# Verify source
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo -e "${RED}✗ llama.cpp source not found at ${LLAMA_CPP_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ llama.cpp source found${NC}"
echo ""

# Build based on target
case "$TARGET" in
    940m)
        build_for_gpu "940m"
        ;;
    t4)
        build_for_gpu "t4"
        ;;
    both)
        build_for_gpu "940m"
        echo ""
        build_for_gpu "t4"
        ;;
    *)
        echo -e "${RED}✗ Unknown target: $TARGET${NC}"
        echo -e "${YELLOW}Use: 940m, t4, both, or auto${NC}"
        exit 1
        ;;
esac

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}✓ All builds completed successfully!${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
