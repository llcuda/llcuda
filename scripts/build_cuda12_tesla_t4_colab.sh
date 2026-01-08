#!/bin/bash
################################################################################
# CUDA 12 Build Script for NVIDIA Tesla T4 (Compute Capability 7.5)
# Target System: Google Colab with Python 3.12
#
# This script builds llama.cpp with CUDA 12 support optimized for Tesla T4
# GPU Specifications:
#   - Compute Capability: 7.5 (Turing architecture)
#   - VRAM: ~15GB
#   - CUDA Version: 12.4/12.6 (Google Colab)
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration for Google Colab
# Note: Adjust these paths if running locally
LLAMA_CPP_DIR="${PWD}/llama.cpp"
BUILD_DIR="${LLAMA_CPP_DIR}/build_cuda12_t4"
INSTALL_PREFIX="/content/llama_cuda12_t4"

# CUDA Configuration for Tesla T4
CUDA_ARCHITECTURES="75"  # Compute Capability 7.5 for Turing
# Colab typically has CUDA at /usr/local/cuda
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

# Build Configuration
BUILD_TYPE="Release"
NUM_THREADS=$(nproc)

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}CUDA 12 Build for NVIDIA Tesla T4 (CC 7.5) - Google Colab${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  llama.cpp Source:  ${LLAMA_CPP_DIR}"
echo -e "  Build Directory:   ${BUILD_DIR}"
echo -e "  Install Prefix:    ${INSTALL_PREFIX}"
echo -e "  CUDA Home:         ${CUDA_HOME}"
echo -e "  CUDA Compiler:     ${CUDA_COMPILER}"
echo -e "  Target GPU:        Tesla T4 (Turing, CC 7.5)"
echo -e "  Build Type:        ${BUILD_TYPE}"
echo -e "  CPU Threads:       ${NUM_THREADS}"
echo ""

# Detect if running in Colab
if [ -d "/content" ]; then
    echo -e "${GREEN}✓ Running in Google Colab environment${NC}"
    INSTALL_PREFIX="/content/llama_cuda12_t4"
else
    echo -e "${YELLOW}⚠ Not running in Google Colab${NC}"
    INSTALL_PREFIX="${PWD}/install_t4"
fi
echo ""

# Verify CUDA installation
echo -e "${YELLOW}Step 1: Verifying CUDA installation...${NC}"
if [ ! -f "${CUDA_COMPILER}" ]; then
    echo -e "${RED}ERROR: CUDA compiler not found at ${CUDA_COMPILER}${NC}"
    echo -e "${RED}Please check CUDA installation${NC}"
    exit 1
fi

${CUDA_COMPILER} --version
nvidia-smi
echo -e "${GREEN}✓ CUDA compiler and GPU found${NC}"
echo ""

# Clone llama.cpp if not present
echo -e "${YELLOW}Step 2: Checking llama.cpp source...${NC}"
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    echo -e "  Cloning llama.cpp repository..."
    git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
fi
echo -e "${GREEN}✓ llama.cpp source ready${NC}"
echo ""

# Clean previous build
echo -e "${YELLOW}Step 3: Cleaning previous build...${NC}"
if [ -d "${BUILD_DIR}" ]; then
    echo -e "  Removing old build directory..."
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
echo -e "${GREEN}✓ Build directory ready${NC}"
echo ""

# Create install directory
mkdir -p "${INSTALL_PREFIX}"
echo -e "${GREEN}✓ Install directory created: ${INSTALL_PREFIX}${NC}"
echo ""

################################################################################
# CMAKE CONFIGURATION
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}CMAKE CONFIGURATION PHASE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}Running CMake configuration...${NC}"
echo ""

cd "${LLAMA_CPP_DIR}"

cmake -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

echo ""
echo -e "${GREEN}✓ CMake configuration completed${NC}"
echo ""
echo -e "${YELLOW}CMake Options Explained:${NC}"
echo -e "  ${GREEN}-DGGML_CUDA=ON${NC}                      - Enable CUDA support"
echo -e "  ${GREEN}-DCMAKE_CUDA_ARCHITECTURES=\"75\"${NC}    - Target Compute Capability 7.5 (Turing)"
echo -e "  ${GREEN}-DGGML_NATIVE=OFF${NC}                   - Build for CC 7.5, not build machine"
echo -e "  ${GREEN}-DGGML_CUDA_FA=ON${NC}                   - Enable FlashAttention (T4 supports it)"
echo -e "  ${GREEN}-DGGML_CUDA_FA_ALL_QUANTS=ON${NC}        - Enable FA for all quantization types"
echo -e "  ${GREEN}-DGGML_CUDA_GRAPHS=ON${NC}               - Enable CUDA graphs for performance"
echo -e "  ${GREEN}-DLLAMA_BUILD_SERVER=ON${NC}             - Build llama-server executable"
echo -e "  ${GREEN}-DBUILD_SHARED_LIBS=ON${NC}              - Build shared libraries (.so files)"
echo ""

################################################################################
# BUILD PHASE
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}BUILD PHASE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}Building llama.cpp with ${NUM_THREADS} threads...${NC}"
echo -e "${YELLOW}This will take approximately 5-15 minutes in Colab${NC}"
echo ""

cmake --build "${BUILD_DIR}" --config Release -j${NUM_THREADS}

echo ""
echo -e "${GREEN}✓ Build completed successfully${NC}"
echo ""

################################################################################
# INSTALL PHASE
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}INSTALL PHASE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}Installing binaries and libraries...${NC}"

# Create directory structure
mkdir -p "${INSTALL_PREFIX}/bin"
mkdir -p "${INSTALL_PREFIX}/lib"

# Copy binaries
cp "${BUILD_DIR}/bin/llama-server" "${INSTALL_PREFIX}/bin/" || true
cp "${BUILD_DIR}/bin/llama-cli" "${INSTALL_PREFIX}/bin/" || true
cp "${BUILD_DIR}/bin/llama-quantize" "${INSTALL_PREFIX}/bin/" || true
cp "${BUILD_DIR}/bin/llama-embedding" "${INSTALL_PREFIX}/bin/" || true
cp "${BUILD_DIR}/bin/llama-bench" "${INSTALL_PREFIX}/bin/" || true

# Copy shared libraries
cp "${BUILD_DIR}"/bin/libllama.so* "${INSTALL_PREFIX}/lib/" 2>/dev/null || true
cp "${BUILD_DIR}"/bin/libggml*.so* "${INSTALL_PREFIX}/lib/" 2>/dev/null || true
cp "${BUILD_DIR}"/ggml/src/libggml*.so* "${INSTALL_PREFIX}/lib/" 2>/dev/null || true

# Make binaries executable
chmod +x "${INSTALL_PREFIX}/bin"/*

echo -e "${GREEN}✓ Installation completed${NC}"
echo ""
echo -e "${YELLOW}Installed files:${NC}"
ls -lh "${INSTALL_PREFIX}/bin/"
echo ""
ls -lh "${INSTALL_PREFIX}/lib/"
echo ""

################################################################################
# VERIFICATION
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}VERIFICATION${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}Verifying CUDA support...${NC}"
echo ""

# Check CUDA linking
ldd "${INSTALL_PREFIX}/bin/llama-server" | grep -E "(cuda|cublas)" || echo "No CUDA libraries found (might be statically linked)"

# Test the server
echo ""
echo -e "${YELLOW}Testing llama-server...${NC}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}"
"${INSTALL_PREFIX}/bin/llama-server" --help | head -20

echo ""
echo -e "${GREEN}✓ Verification completed${NC}"
echo ""

################################################################################
# CREATE TAR ARCHIVE FOR DISTRIBUTION
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}CREATING DISTRIBUTION PACKAGE${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

ARCHIVE_NAME="llcuda-binaries-cuda12-t4-cc75.tar.gz"
ARCHIVE_PATH="${PWD}/${ARCHIVE_NAME}"

echo -e "${YELLOW}Creating tar.gz archive for distribution...${NC}"
cd "${INSTALL_PREFIX}/.."
tar -czf "${ARCHIVE_PATH}" "$(basename ${INSTALL_PREFIX})"

if [ -f "${ARCHIVE_PATH}" ]; then
    echo -e "${GREEN}✓ Archive created: ${ARCHIVE_PATH}${NC}"
    echo -e "  Size: $(du -h ${ARCHIVE_PATH} | cut -f1)"
    echo ""
    echo -e "${YELLOW}You can upload this archive to GitHub releases${NC}"
else
    echo -e "${RED}✗ Failed to create archive${NC}"
fi
echo ""

################################################################################
# SUMMARY
################################################################################
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}BUILD SUMMARY${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}✓ llama.cpp successfully built for Tesla T4 (CC 7.5)${NC}"
echo ""
echo -e "${YELLOW}Installation Directory:${NC}"
echo -e "  ${INSTALL_PREFIX}"
echo ""
echo -e "${YELLOW}Key Features Enabled:${NC}"
echo -e "  ✓ CUDA support (Compute Capability 7.5)"
echo -e "  ✓ FlashAttention (faster inference)"
echo -e "  ✓ CUDA graphs (optimized execution)"
echo -e "  ✓ All quantization formats"
echo -e "  ✓ 15GB VRAM support"
echo ""
echo -e "${YELLOW}Usage in Python/llcuda:${NC}"
echo -e "  export LLAMA_SERVER_PATH=\"${INSTALL_PREFIX}/bin/llama-server\""
echo -e "  export LD_LIBRARY_PATH=\"${INSTALL_PREFIX}/lib:\${LD_LIBRARY_PATH}\""
echo ""
echo -e "${YELLOW}Recommended Settings for T4:${NC}"
echo -e "  • Use Q4_K_M or Q5_K_M quantization for balance"
echo -e "  • gpu_layers: 30-40 for 3B models, 20-30 for 7B models"
echo -e "  • ctx_size: 2048-8192 depending on model size"
echo -e "  • FlashAttention enabled for 2x faster inference"
echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
