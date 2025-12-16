#!/bin/bash

INPUT_DIR="MolJSON"
OUTPUT_DIR='output'
EXECUTABLE_PATH='build/ir_spec'
FAILED_COUNT=0
TOTAL_COUNT=0

if [ ! -d "build/" ]; then
  echo "CRITICAL ERROR: Build directory not found."
  echo "Please run build.sh to compile code for executable"
  echo "Aborting script execution"
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory '$INPUT_DIR' not found."
  exit 1
fi

echo "Starting Vibration Calculations..."

# case if an argument is supplied for a single execution
if [ "$#" -eq 1 ]; then

  input_file="$1"
  if [ -f "$input_file" ]; then

    filename=$(basename -- "$input_file")

    filename_no_ext="${filename%.*}"

    output_file="$OUTPUT_DIR/$filename_no_ext.out"

    echo "Calculating Frequencies of: $filename_no_ext -> writing to $output_file"

    "$EXECUTABLE_PATH" "$input_file" >"$output_file"
    EXIT_STATUS=$?

    if [ $EXIT_STATUS -ne 0 ]; then
      echo "--- FAILED: Executable failed for file '$input_file'"
      ((FAILED_COUNT++))
    else
      echo "--- SUCCESS: Finished processing '$input_file'"
    fi

  fi
  exit 0
fi

# By default, run through all input files
for input_file in "$INPUT_DIR"/*; do
  if [ -f "$input_file" ]; then
    ((TOTAL_COUNT++))

    filename=$(basename -- "$input_file")

    filename_no_ext="${filename%.*}"

    output_file="$OUTPUT_DIR/$filename_no_ext.out"

    echo "Calculating Frequencies of: $filename_no_ext -> writing to $output_file"

    "$EXECUTABLE_PATH" "$input_file" >"$output_file"
    EXIT_STATUS=$?

    if [ $EXIT_STATUS -ne 0 ]; then
      echo "--- FAILED: Executable failed for file '$input_file'"
      ((FAILED_COUNT++))
    else
      echo "--- SUCCESS: Finished processing '$input_file'"
    fi

  fi
done

echo
echo "--------------------------------------------"
echo
echo "Finished Processing files!"
