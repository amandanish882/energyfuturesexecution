cmake_minimum_required(VERSION 3.14)
project(pricing_kernel CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Debug build
set(CMAKE_BUILD_TYPE Debug)

# Find pybind11 (installed via pip or system package)
find_package(pybind11 REQUIRED)

# Build the Python extension module
pybind11_add_module(pricing_kernel
    bindings/pybind_module.cpp
)

# Add the include directory so headers are found
target_include_directories(pricing_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
