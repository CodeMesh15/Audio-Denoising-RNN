#!/bin/bash

# This script provides instructions for downloading the LibriSpeech (clean)
# and ESC-50 (noise) datasets for the Audio-Denoising-RNN project.

# Create directories for the raw data
mkdir -p raw_data/clean
mkdir -p raw_data/noise

echo "--- Audio Denoising Data Download Guide ---"
echo ""
echo "This script will not download the data automatically."
echo "Please follow these manual steps:"
echo ""
echo "1. Download the LibriSpeech dataset (dev-clean.tar.gz is a good start):"
echo "   Go to: https://www.openslr.org/12"
echo "   Download 'dev-clean.tar.gz' [337M]"
echo "   Extract it and move the contents into 'data/raw_data/clean/'"
echo ""
echo "2. Download the ESC-50 dataset for environmental noise:"
echo "   Go to: https://github.com/karolpiczak/ESC-50"
echo "   Download the zip file."
echo "   Extract it and move the 'audio' folder's contents into 'data/raw_data/noise/'"
echo ""
echo "After these files are in place, you can run the prepare_dataset.py script."
echo ""
echo "------------------------------------------"
