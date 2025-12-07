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

# If Needed Make Modifications below this line 

add_executable(ir_spec src/ir_spectrum.cpp)

target_include_directories(ir_spec PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/basis_functions
    ${CMAKE_SOURCE_DIR}/include/fock
    ${CMAKE_SOURCE_DIR}/include/gradient
    ${CMAKE_SOURCE_DIR}/include/nr
    ${CMAKE_SOURCE_DIR}/include/geometry_optimization
)

add_executable(steepest_descent src/steepest_descent.cpp)

target_include_directories(steepest_descent PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/basis_functions
    ${CMAKE_SOURCE_DIR}/include/fock
    ${CMAKE_SOURCE_DIR}/include/gradient
    ${CMAKE_SOURCE_DIR}/include/nr
    ${CMAKE_SOURCE_DIR}/include/geometry_optimization
)

# Python bindings for steepest descent (if pybind11 is found)
if(pybind11_FOUND)
    # Create an object library to avoid recompiling steepest_descent.cpp
    # and to share symbols without duplicating them
    add_library(steepest_descent_obj OBJECT src/steepest_descent.cpp)
    target_include_directories(steepest_descent_obj PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/basis_functions
        ${CMAKE_SOURCE_DIR}/include/fock
        ${CMAKE_SOURCE_DIR}/include/gradient
        ${CMAKE_SOURCE_DIR}/include/nr
        ${CMAKE_SOURCE_DIR}/include/geometry_optimization
    )
    target_compile_definitions(steepest_descent_obj PRIVATE NO_MAIN)
    
    # Create Python module using the object library
    pybind11_add_module(steepest_descent_py src/steepest_descent_bindings.cpp $<TARGET_OBJECTS:steepest_descent_obj>)
    target_include_directories(steepest_descent_py PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/basis_functions
        ${CMAKE_SOURCE_DIR}/include/fock
        ${CMAKE_SOURCE_DIR}/include/gradient
        ${CMAKE_SOURCE_DIR}/include/nr
        ${CMAKE_SOURCE_DIR}/include/geometry_optimization
    )
    
    message(STATUS "pybind11 found - Python bindings will be built")
else()
    message(STATUS "pybind11 not found - Python bindings will not be built")
endif()
