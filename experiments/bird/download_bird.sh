#!/bin/bash
set -e

OUTPUT_DIR="${1:-./data/bird}"

if [ -d "$OUTPUT_DIR/dev" ]; then
    echo "BIRD dataset already exists"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Downloading BIRD dataset..."

if command -v gdown &> /dev/null; then
    gdown "https://drive.google.com/uc?id=1cGqTKxF3vgvB1VEQ0cYA-e4GVKDXHNH_" -O dev.zip
else
    echo "Error: gdown not installed"
    echo "Install: pip install gdown"
    exit 1
fi

echo "Extracting..."
unzip -q dev.zip
rm dev.zip

echo "Done"
echo "Location: $OUTPUT_DIR"
