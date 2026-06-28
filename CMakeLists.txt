cmake_minimum_required(VERSION 3.15)
project(SovereignGrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library source files
set(LIB_SOURCES
    src/allocator.cpp
    src/tensor.cpp
    src/autograd.cpp
    src/ops.cpp
)

set(CUDA_SOURCES
    src/cuda_kernels.cu
)

# Try to find CUDA
option(USE_CUDA "Use physical NVIDIA CUDA hardware if available" ON)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}. Building with physical GPU support.")
        add_definitions(-DSOVEREIGNGRAD_USE_CUDA)
        set(WITH_CUDA_HARDWARE ON)
    else()
        message(STATUS "CUDA compiler not found. Building with CPU-based CUDA Emulator.")
        set(WITH_CUDA_HARDWARE OFF)
    endif()
else()
    message(STATUS "CUDA disabled by user option. Building with CPU-based CUDA Emulator.")
    set(WITH_CUDA_HARDWARE OFF)
endif()

# Core static library target
add_library(sovereign_grad_lib STATIC ${LIB_SOURCES} ${CUDA_SOURCES})

if(WITH_CUDA_HARDWARE)
    set_target_properties(sovereign_grad_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    # CUDA Mock/Emulation build - compile .cu files as standard C++ files
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES COMPILE_FLAGS "-x c++")
    target_compile_definitions(sovereign_grad_lib PRIVATE SOVEREIGNGRAD_CUDA_EMULATOR)
endif()

target_include_directories(sovereign_grad_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler flags for GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sovereign_grad_lib PRIVATE -Wall -Wextra -O3 -g)
endif()

# Executable: Core unit tests and synthetic training demo
add_executable(sovereign_grad src/main.cpp)
target_link_libraries(sovereign_grad PRIVATE sovereign_grad_lib)

# Executable: Comparative Benchmark Suite
add_executable(sovereign_grad_benchmark src/benchmark.cpp)
target_link_libraries(sovereign_grad_benchmark PRIVATE sovereign_grad_lib)

# Executable: MNIST digit convolutional network classifier
add_executable(mnist_demo src/mnist.cpp src/mnist_demo.cpp)
target_link_libraries(mnist_demo PRIVATE sovereign_grad_lib)

# CMake Packaging & Installation exports
include(GNUInstallDirs)
install(TARGETS sovereign_grad_lib sovereign_grad sovereign_grad_benchmark mnist_demo
    EXPORT SovereignGradTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/soverigngrad DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SovereignGradTargets
    FILE SovereignGradTargets.cmake
    NAMESPACE SovereignGrad::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SovereignGrad
)

# Create standard package configuration file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/SovereignGradConfig.cmake"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/SovereignGradTargets.cmake\")\n"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SovereignGradConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SovereignGrad
)

