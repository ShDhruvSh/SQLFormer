#!/bin/bash
#
# Download BIRD Dataset
#
# BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation)
# is a challenging text-to-SQL benchmark with dirty data and external knowledge.
#
# Usage: ./download_bird.sh [output_dir]
#

set -e

OUTPUT_DIR="${1:-./data/bird}"

echo "=============================================="
echo "BIRD Dataset Downloader"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Check if already downloaded
if [ -d "dev" ] && [ -f "dev/dev.json" ]; then
    echo "BIRD dev set already exists. Skipping download."
    echo "To re-download, remove the dev/ directory first."
else
    echo "Downloading BIRD development set..."

    # BIRD is hosted on Google Drive - use gdown if available
    if command -v gdown &> /dev/null; then
        echo "Using gdown to download..."
        # Dev set file ID (from BIRD repository)
        gdown "https://drive.google.com/uc?id=1cGqTKxF3vgvB1VEQ0cYA-e4GVKDXHNH_" -O dev.zip
    else
        echo ""
        echo "ERROR: gdown not installed."
        echo ""
        echo "Please install gdown first:"
        echo "  pip install gdown"
        echo ""
        echo "Or download manually from:"
        echo "  https://bird-bench.github.io/"
        echo ""
        echo "After downloading dev.zip, extract it to:"
        echo "  $OUTPUT_DIR/dev/"
        exit 1
    fi

    echo "Extracting dev.zip..."
    unzip -q dev.zip
    rm dev.zip
    echo "Done!"
fi

# Verify structure
echo ""
echo "Verifying dataset structure..."

if [ -f "dev/dev.json" ]; then
    EXAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('dev/dev.json'))))")
    echo "  ✓ dev.json found ($EXAMPLE_COUNT examples)"
else
    echo "  ✗ dev.json not found!"
    exit 1
fi

if [ -d "dev/dev_databases" ]; then
    DB_COUNT=$(ls -d dev/dev_databases/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ dev_databases/ found ($DB_COUNT databases)"
else
    echo "  ✗ dev_databases/ not found!"
    exit 1
fi

echo ""
echo "=============================================="
echo "BIRD Dataset Ready!"
echo "=============================================="
echo ""
echo "Dataset location: $OUTPUT_DIR"
echo ""
echo "To run experiments:"
echo ""
echo "  cd experiments/bird"
echo "  python run_experiment.py \\"
echo "      --data $OUTPUT_DIR/dev/dev.json \\"
echo "      --database-dir $OUTPUT_DIR/dev/dev_databases \\"
echo "      --methods hybrid \\"
echo "      --max-examples 10"
echo ""
echo "To evaluate:"
echo ""
echo "  python evaluate.py \\"
echo "      --predictions predictions/hybrid_predictions.json \\"
echo "      --data $OUTPUT_DIR/dev/dev.json \\"
echo "      --database-dir $OUTPUT_DIR/dev/dev_databases \\"
echo "      --model-name hybrid"
echo ""
