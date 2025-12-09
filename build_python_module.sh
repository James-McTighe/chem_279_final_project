#!/bin/bash

# Build script for steepest descent Python module


python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    pip install pybind11
fi

st
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Get pybind11 CMake directory
PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)

# Run CMake
if [ -n "$PYBIND11_DIR" ]; then
    cmake -Dpybind11_DIR="$PYBIND11_DIR" .. > /dev/null 2>&1
else
    cmake .. > /dev/null 2>&1
fi

# Build the project
make steepest_descent_py -j$(nproc) > /dev/null 2>&1

# Check if build was successful
if ls steepest_descent_py*.so 1> /dev/null 2>&1 || ls steepest_descent_py*.dylib 1> /dev/null 2>&1; then
    echo "Python module build successful!"
    
    # Optionally install to site-packages
    if [ "$1" == "--install" ]; then
        SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
        cp steepest_descent_py*.so "$SITE_PACKAGES/" 2>/dev/null || cp steepest_descent_py*.dylib "$SITE_PACKAGES/" 2>/dev/null
        echo "Module installed to site-packages: $SITE_PACKAGES"
    fi
fi
