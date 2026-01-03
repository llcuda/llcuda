# CMakeLists.txt for Python bindings
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# If this is being built as part of pip install
if(NOT DEFINED BUILD_PYTHON_BINDINGS)
    set(BUILD_PYTHON_BINDINGS OFF)
endif()

if(BUILD_PYTHON_BINDINGS)
    # Python bindings build configuration
    project(llcuda_python LANGUAGES CXX CUDA)
    
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Find pybind11
    find_package(pybind11 CONFIG REQUIRED)
    
    # Find CUDA
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA architecture (default to T4 for Kaggle/Colab)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architecture")
    endif()
    
    message(STATUS "Python bindings build configuration:")
    message(STATUS "  Python: ${Python3_VERSION}")
    message(STATUS "  Python include: ${Python3_INCLUDE_DIRS}")
    message(STATUS "  pybind11: ${pybind11_VERSION}")
    message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # Include parent CMakeLists.txt to build the main library
    # But we'll build it separately in the parent directory
    
    # Add parent directory to include path
    include_directories(${CMAKE_SOURCE_DIR}/include)
    
    # Create interface library for headers
    add_library(llcuda_headers_py INTERFACE)
    target_include_directories(llcuda_headers_py INTERFACE
        ${CMAKE_SOURCE_DIR}/include
    )
    
    # Build core libraries (simplified for Python package)
    
    # Core library
    add_library(llcuda_core_py STATIC
        ${CMAKE_SOURCE_DIR}/src/core/inference_engine.cpp
        ${CMAKE_SOURCE_DIR}/src/core/model_manager.cpp
        ${CMAKE_SOURCE_DIR}/src/core/request_processor.cpp
        ${CMAKE_SOURCE_DIR}/src/core/metrics_collector.cpp
        ${CMAKE_SOURCE_DIR}/src/core/http_client.cpp
    )
    target_link_libraries(llcuda_core_py PUBLIC llcuda_headers_py Threads::Threads)
    
    # CUDA backend
    add_library(llcuda_cuda_py STATIC
        ${CMAKE_SOURCE_DIR}/src/cuda/cuda_backend.cu
        ${CMAKE_SOURCE_DIR}/src/cuda/custom_kernels.cu
        ${CMAKE_SOURCE_DIR}/src/cuda/memory_manager.cu
        ${CMAKE_SOURCE_DIR}/src/cuda/stream_manager.cu
    )
    target_link_libraries(llcuda_cuda_py PUBLIC
        llcuda_headers_py
        CUDA::cudart
    )
    set_target_properties(llcuda_cuda_py PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Storage library
    add_library(llcuda_storage_py STATIC
        ${CMAKE_SOURCE_DIR}/src/storage/content_addressed_store.cpp
        ${CMAKE_SOURCE_DIR}/src/storage/manifest_manager.cpp
        ${CMAKE_SOURCE_DIR}/src/storage/cache_manager.cpp
        ${CMAKE_SOURCE_DIR}/src/storage/storage_client.cpp
        ${CMAKE_SOURCE_DIR}/src/storage/sha256.cpp
    )
    target_link_libraries(llcuda_storage_py PUBLIC llcuda_headers_py)
    
    # Python bindings module
    pybind11_add_module(_llcuda 
        ${CMAKE_SOURCE_DIR}/python/llcuda_py.cpp
    )
    
    target_link_libraries(_llcuda PRIVATE
        llcuda_core_py
        llcuda_cuda_py
        llcuda_storage_py
        CUDA::cudart
    )
    
    # Set output name
    set_target_properties(_llcuda PROPERTIES
        OUTPUT_NAME "_llcuda"
        PREFIX ""
    )
    
    # Installation
    install(TARGETS _llcuda
        LIBRARY DESTINATION llcuda
    )
    
else()
    # This is the main project build - include original CMakeLists.txt
    # (Already handled by parent)
endif()
