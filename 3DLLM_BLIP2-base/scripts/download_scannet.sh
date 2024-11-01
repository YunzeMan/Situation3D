#!/bin/bash

# Define the base directory
BASE_DIR="/u/yunzem2/links/bbsg/dataset/ScanNet/scans"

# Define the pattern for the .ply files
PATTERN="scene*_vh_clean_2.ply"

# Define the name of the output zip file
OUTPUT_ZIP="ply_files_backup.zip"

# Navigate to the base directory
cd "$BASE_DIR"

# Find and zip the files
find . -name "$PATTERN" -exec zip "$OUTPUT_ZIP" '{}' +

echo "Zip file created: $OUTPUT_ZIP"
