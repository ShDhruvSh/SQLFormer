#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SPIDER_DIR="${DATA_DIR}/spider"

if [ -f "${SPIDER_DIR}/dev.json" ]; then
    echo "Spider dataset already exists"
    exit 0
fi

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "Downloading Spider dataset..."

if command -v gdown &> /dev/null; then
    gdown "1TqleXec_OykOYFREKKtschzY29dUcVAQ" -O spider.zip
else
    echo "Error: gdown not installed"
    echo "Install: pip install gdown"
    exit 1
fi

if [ ! -f "spider.zip" ]; then
    echo "Download failed"
    exit 1
fi

echo "Extracting..."
unzip -q spider.zip
rm spider.zip

if [ -d "spider_data" ]; then
    mv spider_data spider
fi

echo "Done"
echo "Location: ${SPIDER_DIR}"
