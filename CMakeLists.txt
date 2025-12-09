cmake_minimum_required (VERSION 3.31)

project (homework_4 LANGUAGES CXX)

# Compiler requirements
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find JSON Libraries
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Find pybind11 for Python bindings
find_package(pybind11 CONFIG)

add_library(common INTERFACE)

target_link_libraries(common INTERFACE
    Eigen3::Eigen
    armadillo
    nlohmann_json::nlohmann_json
    HDF5::HDF5
)

target_compile_definitions(common INTERFACE 
    ARMA_USE_HDF5
    H5_USE_110_API
)

link_libraries(common)

# Build options - default to ON
option(BUILD_IR_SPEC "Build ir_spec executable" ON)
option(BUILD_STEEPEST_DESCENT "Build steepest_descent executable" ON)
option(BUILD_PYTHON_MODULE "Build Python module" ON)

# Check if user explicitly specified any build options
if(DEFINED BUILD_IR_SPEC OR DEFINED BUILD_STEEPEST_DESCENT OR DEFINED BUILD_PYTHON_MODULE)
    # User specified at least one option, so set unspecified ones to OFF
    if(NOT DEFINED BUILD_IR_SPEC)
        set(BUILD_IR_SPEC OFF)
    endif()
    if(NOT DEFINED BUILD_STEEPEST_DESCENT)
        set(BUILD_STEEPEST_DESCENT OFF)
    endif()
    if(NOT DEFINED BUILD_PYTHON_MODULE)
        set(BUILD_PYTHON_MODULE OFF)
    endif()
endif()

# If Needed Make Modifications below this line 

if(BUILD_IR_SPEC)
    add_executable(ir_spec src/ir_spectrum.cpp)

    target_include_directories(ir_spec PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/basis_functions
        ${CMAKE_SOURCE_DIR}/include/fock
        ${CMAKE_SOURCE_DIR}/include/gradient
        ${CMAKE_SOURCE_DIR}/include/nr
        ${CMAKE_SOURCE_DIR}/include/geometry_optimization
    )
    message(STATUS "Building ir_spec executable")
endif()

if(BUILD_STEEPEST_DESCENT)
    add_executable(steepest_descent src/steepest_descent.cpp)

    target_include_directories(steepest_descent PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/basis_functions
        ${CMAKE_SOURCE_DIR}/include/fock
        ${CMAKE_SOURCE_DIR}/include/gradient
        ${CMAKE_SOURCE_DIR}/include/nr
        ${CMAKE_SOURCE_DIR}/include/geometry_optimization
    )
    message(STATUS "Building steepest_descent executable")
endif()

# Python bindings for steepest descent (if pybind11 is found and BUILD_PYTHON_MODULE is ON)
if(BUILD_PYTHON_MODULE AND pybind11_FOUND)
    # Create the Python module by compiling both the bindings and steepest_descent.cpp
    # but define NO_MAIN for steepest_descent to exclude main()
    pybind11_add_module(steepest_descent_py 
        src/steepest_descent_bindings.cpp 
        src/steepest_descent.cpp
    )
    target_include_directories(steepest_descent_py PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/basis_functions
        ${CMAKE_SOURCE_DIR}/include/fock
        ${CMAKE_SOURCE_DIR}/include/gradient
        ${CMAKE_SOURCE_DIR}/include/nr
        ${CMAKE_SOURCE_DIR}/include/geometry_optimization
    )
    target_compile_definitions(steepest_descent_py PRIVATE NO_MAIN)
    
    message(STATUS "Building Python module steepest_descent_py")
elseif(BUILD_PYTHON_MODULE AND NOT pybind11_FOUND)
    message(WARNING "BUILD_PYTHON_MODULE is ON but pybind11 not found - Python bindings will not be built")
endif()
