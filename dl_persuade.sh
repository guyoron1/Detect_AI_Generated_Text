#!/bin/bash

# Define the directory structure
BASE_DIR="external_sources"
PERSUADE_DIR="$BASE_DIR/persuade"

# Ensure the directories exist
mkdir -p "$PERSUADE_DIR"

# Download files into the Transformer directory
FILES=(
    "https://www.dropbox.com/scl/fi/1jx5959whlgt2qket3mgd/persuade_corpus_2.0_train.csv?rlkey=tn4ol05y3k7csx4xb612b3e7u&st=xnopy8lk&dl=1"
)
for url in "${FILES[@]}"; do
    # Extract the filename by removing the query string
    filename=$(basename "${url%%\?*}")
    wget -O "$PERSUADE_DIR/$filename" "$url"
done


echo "Setup completed successfully."
