#!/bin/bash

DATASET_URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
OUTPUT_FILE="ml-latest-small.zip"
TARGET_DIR="ml-latest-small"

echo "Downloading MovieLens latest small ~100,000 ratings"
curl -o $OUTPUT_FILE $DATASET_URL

if [ $? -eq 0 ]; then
    echo "Download completed successfully."

    echo "Unzipping the dataset..."
    unzip $OUTPUT_FILE 

    mv $TARGET_DIR data/
    rm $OUTPUT_FILE 
else
    echo "Failed to download the dataset."
fi