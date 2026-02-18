#!/bin/bash

# docs.sh
# Builds the MkDocs site and pushes to the gh-pages branch.

set -e # Exit on error

# 0. Resolve Script Directory
# This ensures the script runs correctly regardless of CWD.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "--- Golem Documentation Deployment ---"
echo "Working Directory: $PWD"

# 1. Check for Virtual Environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: Not running inside a virtual environment."
    echo "Recommend running: source .venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Install Docs Dependencies
echo "Installing/Verifying documentation dependencies..."
pip install mkdocs mkdocs-material

# 3. Clean previous builds
echo "Cleaning previous builds..."
rm -rf site/

# 4. Build and Deploy
# The 'gh-deploy' command builds the 'docs/' directory into HTML 
# and pushes it to the 'gh-pages' branch of the origin remote.
echo "Deploying to GitHub Pages..."
mkdocs gh-deploy --force

echo "--- Deployment Complete ---"
echo "Site should be live shortly at: https://chinchalinchin.github.io/golem/"