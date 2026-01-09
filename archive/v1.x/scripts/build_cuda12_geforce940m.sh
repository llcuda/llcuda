#!/bin/bash
################################################################################
# CUDA 12 Build Script for NVIDIA GeForce 940M (Compute Capability 5.0)
# Target System: Xubuntu 22.04 with Python 3.11
#
# This script builds llama.cpp with CUDA 12 support optimized for GeForce 940M
# GPU Specifications:
#   - Compute Capability: 5.0 (Maxwell architecture)
#   - VRAM: ~1GB
#   - CUDA Version: 12.8 (from system)
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/media/waqasm86/External1/Project-Nvidia"
LLAMA_CPP_DIR="${PROJECT_DIR}/llama.cpp"
BUILD_DIR="${LLAMA_CPP_DIR}/build_cuda12_940m"
INSTALL_DIR="${PROJECT_DIR}/llcuda/llcuda/binaries/cuda12_940m"
LIB_INSTALL_DIR="${PROJECT_DIR}/llcuda/llcuda/lib"

# CUDA Configuration for GeForce 940M
CUDA_ARCHITECTURES="50"  # Compute Capability 5.0 for Maxwell
CUDA_HOME="/usr/local/cuda-12.8"
CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

# Build Configuration
BUILD_TYPE="Release"
NUM_THREADS=$(nproc)

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}CUDA 12 Build for NVIDIA GeForce 940M (CC 5.0)${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Project Directory: ${PROJECT_DIR}"
echo -e "  llama.cpp Source:  ${LLAMA_CPP_DIR}"
echo -e "  Build Directory:   ${BUILD_DIR}"
echo -e "  Install Directory: ${INSTALL_DIR}"
echo -e "  CUDA Home:         ${CUDA_HOME}"
echo -e "  CUDA Compiler:     ${CUDA_COMPILER}"
echo -e "  Target GPU:        GeForce 940M (Maxwell, CC 5.0)"
echo -e "  Build Type:        ${BUILD_TYPE}"
echo -e "  CPU Threads:       ${NUM_THREADS}"
echo ""

# Verify CUDA installation
echo -e "${YELLOW}Step 1: Verifying CUDA installation...${NC}"
if [ ! -f "${CUDA_COMPILER}" ]; then
    echo -e "${RED}ERROR: CUDA compiler not found at ${CUDA_COMPILER}${NC}"
    echo -e "${RED}Please install CUDA 12.8 or update CUDA_HOME variable${NC}"
    exit 1
fi

${CUDA_COMPILER} --version
echo -e "${GREEN}✓ CUDA compiler found${NC}"
echo ""

# Verify source directory
echo -e "${YELLOW}Step 2: Verifying llama.cpp source...${NC}"
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo -e "${RED}ERROR: llama.cpp source not found at ${LLAMA_CPP_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ llama.cpp source found${NC}"
echo ""

# Clean previous build (optional)
echo -e "${YELLOW}Step 3: Cleaning previous build...${NC}"
if [ -d "${BUILD_DIR}" ]; then
    echo -e "  Removing old build directory..."
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
echo -e "${GREEN}✓ Build directory ready${NC}"
echo ""

# Create install directories
echo -e "${YELLOW}Step 4: Creating install directories...${NC}"
mkdir -p "${INSTALL_DIR}"
mkdir -p "${LIB_INSTALL_DIR}"
echo -e "${GREEN}✓ Install directories created${NC}"
echo ""

################################################################################
# CMAKE CONFIGURATION
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}CMAKE CONFIGURATION PHASE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}CMake Command:${NC}"
echo ""
cat << 'EOF'
cmake -B build_cuda12_940m \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="50" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FA=OFF \
    -DGGML_CUDA_GRAPHS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/.." \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
EOF
echo ""
echo -e "${RED}IMPORTANT: You must run the above command manually in the llama.cpp directory${NC}"
echo -e "${RED}Directory: ${LLAMA_CPP_DIR}${NC}"
echo ""
echo -e "${YELLOW}Explanation of CMake Options:${NC}"
echo -e "  ${GREEN}-DGGML_CUDA=ON${NC}                     - Enable CUDA support"
echo -e "  ${GREEN}-DCMAKE_CUDA_ARCHITECTURES=\"50\"${NC}   - Target Compute Capability 5.0 (Maxwell)"
echo -e "  ${GREEN}-DGGML_NATIVE=OFF${NC}                  - Build for CC 5.0, not build machine GPU"
echo -e "  ${GREEN}-DGGML_CUDA_FORCE_CUBLAS=ON${NC}        - Use cuBLAS (more reliable for older GPUs)"
echo -e "  ${GREEN}-DGGML_CUDA_FA=OFF${NC}                 - Disable FlashAttention (not supported on CC 5.0)"
echo -e "  ${GREEN}-DGGML_CUDA_GRAPHS=ON${NC}              - Enable CUDA graphs for optimization"
echo -e "  ${GREEN}-DLLAMA_BUILD_SERVER=ON${NC}            - Build llama-server executable"
echo -e "  ${GREEN}-DBUILD_SHARED_LIBS=ON${NC}             - Build shared libraries (.so files)"
echo -e "  ${GREEN}-DCMAKE_INSTALL_RPATH=...${NC}          - Set runtime library search path"
echo ""

################################################################################
# BUILD PHASE
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}BUILD PHASE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}Build Command:${NC}"
echo ""
echo "cmake --build build_cuda12_940m --config Release -j${NUM_THREADS}"
echo ""
echo -e "${RED}IMPORTANT: After running CMake configuration, run the above build command${NC}"
echo -e "${RED}Directory: ${LLAMA_CPP_DIR}${NC}"
echo ""
echo -e "${YELLOW}Explanation:${NC}"
echo -e "  ${GREEN}--config Release${NC}    - Build in Release mode (optimized)"
echo -e "  ${GREEN}-j${NUM_THREADS}${NC}               - Use ${NUM_THREADS} parallel build threads"
echo ""

################################################################################
# INSTALL PHASE
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}INSTALL PHASE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}After building, copy the binaries and libraries:${NC}"
echo ""
cat << EOF
# Copy binaries
cp build_cuda12_940m/bin/llama-server "${INSTALL_DIR}/"
cp build_cuda12_940m/bin/llama-cli "${INSTALL_DIR}/"
cp build_cuda12_940m/bin/llama-quantize "${INSTALL_DIR}/"
cp build_cuda12_940m/bin/llama-embedding "${INSTALL_DIR}/"

# Copy shared libraries
cp build_cuda12_940m/bin/libllama.so "${LIB_INSTALL_DIR}/"
cp build_cuda12_940m/bin/libggml*.so "${LIB_INSTALL_DIR}/"

# Make binaries executable
chmod +x "${INSTALL_DIR}"/*

# Verify installation
ls -lh "${INSTALL_DIR}"/
ls -lh "${LIB_INSTALL_DIR}"/
EOF
echo ""

################################################################################
# VERIFICATION
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}VERIFICATION${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}After installation, verify the build:${NC}"
echo ""
cat << 'EOF'
# Check CUDA support
ldd build_cuda12_940m/bin/llama-server | grep cuda

# Test the server (requires a model)
export LD_LIBRARY_PATH="${LIB_INSTALL_DIR}:${LD_LIBRARY_PATH}"
./build_cuda12_940m/bin/llama-server --help
EOF
echo ""

################################################################################
# SUMMARY
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}This script guides you through building llama.cpp for GeForce 940M${NC}"
echo ""
echo -e "${YELLOW}Key Points for GeForce 940M (CC 5.0):${NC}"
echo -e "  • Limited to 1GB VRAM - use small models and low GPU layers"
echo -e "  • FlashAttention disabled (requires CC >= 7.0)"
echo -e "  • cuBLAS forced for better compatibility"
echo -e "  • CUDA graphs enabled for performance"
echo -e "  • Recommended: 4-bit quantized models (Q4_K_M)"
echo -e "  • Suggested gpu_layers: 10-15 for small models"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. cd ${LLAMA_CPP_DIR}"
echo -e "  2. Run the CMake configuration command shown above"
echo -e "  3. Run the build command (this will take 10-30 minutes)"
echo -e "  4. Copy binaries and libraries as shown above"
echo -e "  5. Test with llcuda package"
echo ""
echo -e "${GREEN}Build script completed!${NC}"
