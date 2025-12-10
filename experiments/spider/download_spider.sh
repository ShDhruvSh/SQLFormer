#!/bin/bash
# Download Spider Dataset for SQLFormer Experiments
# Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
# Semantic Parsing and Text-to-SQL Task

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SPIDER_DIR="${DATA_DIR}/spider"

echo "======================================"
echo "  Spider Dataset Download Script"
echo "======================================"

# Create data directory
mkdir -p "${DATA_DIR}"

# Check if already downloaded
if [ -f "${SPIDER_DIR}/dev.json" ] && [ -f "${SPIDER_DIR}/tables.json" ]; then
    echo "Spider dataset already exists at ${SPIDER_DIR}"
    echo "Delete the directory to re-download."
    exit 0
fi

echo ""
echo "Downloading Spider dataset..."
echo "Source: https://yale-lily.github.io/spider"
echo ""

cd "${DATA_DIR}"

# Download Spider dataset
# Note: The official download link may require manual download
# Alternative: Use the GitHub release

SPIDER_URL="https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"

# Try wget first, then curl
if command -v gdown &> /dev/null; then
    echo "Using gdown to download..."
    gdown "1TqleXec_OykOYFREKKtschzY29dUcVAQ" -O spider.zip
elif command -v wget &> /dev/null; then
    echo "Attempting wget download..."
    echo ""
    echo "NOTE: Google Drive downloads may require manual intervention."
    echo "If this fails, please download manually from:"
    echo "  https://yale-lily.github.io/spider"
    echo ""
    wget --no-check-certificate "${SPIDER_URL}" -O spider.zip || {
        echo ""
        echo "Automatic download failed."
        echo ""
        echo "Please download Spider manually:"
        echo "1. Go to: https://yale-lily.github.io/spider"
        echo "2. Click 'Download Spider Dataset'"
        echo "3. Save as: ${DATA_DIR}/spider.zip"
        echo "4. Run this script again"
        exit 1
    }
else
    echo "Neither gdown nor wget found."
    echo ""
    echo "Option 1: Install gdown (recommended)"
    echo "  pip install gdown"
    echo ""
    echo "Option 2: Manual download"
    echo "  1. Go to: https://yale-lily.github.io/spider"
    echo "  2. Click 'Download Spider Dataset'"
    echo "  3. Save as: ${DATA_DIR}/spider.zip"
    echo "  4. Run this script again"
    exit 1
fi

# Check if download succeeded
if [ ! -f "spider.zip" ]; then
    echo "Download failed. Please download manually."
    exit 1
fi

echo ""
echo "Extracting dataset..."
unzip -q spider.zip
rm spider.zip

# Rename directory if needed (Spider sometimes extracts to different names)
if [ -d "spider_data" ]; then
    mv spider_data spider
fi

# Verify structure
echo ""
echo "Verifying dataset structure..."

REQUIRED_FILES=(
    "${SPIDER_DIR}/dev.json"
    "${SPIDER_DIR}/train_spider.json"
    "${SPIDER_DIR}/tables.json"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  MISSING: $file"
        MISSING=1
    else
        echo "  OK: $(basename $file)"
    fi
done

if [ -d "${SPIDER_DIR}/database" ]; then
    DB_COUNT=$(ls -d ${SPIDER_DIR}/database/*/ 2>/dev/null | wc -l)
    echo "  OK: database/ (${DB_COUNT} databases)"
else
    echo "  MISSING: database/"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "WARNING: Some files are missing. Dataset may be incomplete."
    exit 1
fi

echo ""
echo "======================================"
echo "  Download Complete!"
echo "======================================"
echo ""
echo "Dataset location: ${SPIDER_DIR}"
echo ""
echo "Files:"
echo "  - dev.json         : Development set (1034 examples)"
echo "  - train_spider.json: Training set (7000 examples)"
echo "  - tables.json      : Database schemas"
echo "  - database/        : SQLite databases"
echo ""
echo "Next steps:"
echo "  python run_experiment.py \\"
echo "      --tables ${SPIDER_DIR}/tables.json \\"
echo "      --data ${SPIDER_DIR}/dev.json \\"
echo "      --max-examples 100"
