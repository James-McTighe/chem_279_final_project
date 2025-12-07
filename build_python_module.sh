#!/bin/bash

# Build script for steepest descent Python module

echo "Building steepest descent Python module..."

# Check if pybind11 is installed
python3 -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "pybind11 is not installed. Installing..."
    pip install pybind11
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Run CMake
echo "Running CMake..."
cmake ..

# Build the project
echo "Building..."
make steepest_descent_py -j$(nproc)

# Check if build was successful
if [ -f "steepest_descent_py*.so" ] || [ -f "steepest_descent_py*.dylib" ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "The Python module is located in the build directory."
    echo "You can now use it in Python by adding the build directory to your PYTHONPATH:"
    echo ""
    echo "  import sys"
    echo "  sys.path.insert(0, 'build')"
    echo "  import steepest_descent_py"
    echo ""
    echo "Or you can run the example Jupyter notebook: steepest_descent_example.ipynb"
else
    echo ""
    echo "✗ Build failed. Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  1. pybind11 not installed: pip install pybind11"
    echo "  2. Missing dependencies: Make sure all C++ dependencies are installed"
    echo "  3. CMake can't find pybind11: Try 'pip install pybind11[global]'"
fi
