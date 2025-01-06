#!/bin/bash

# Define the repository URL and folder to clone
REPO_URL="https://github.com/ryuryukke/OUTFOX.git"
FOLDER="data"
TARGET_DIR="external_sources/OUTFOX"

# Create the target directory structure if it doesn't exist
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Initialize a new Git repository
git init

# Set the sparse checkout configuration
git config core.sparseCheckout true

# Specify the folder to fetch
echo "$FOLDER/" > .git/info/sparse-checkout

# Fetch the repository
git remote add origin "$REPO_URL"
git pull origin main

# Done
echo "Cloned the '$FOLDER' folder into $TARGET_DIR."
