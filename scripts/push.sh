#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: No commit message provided."
    echo "Usage: ./push.sh \"Your commit message\""
    exit 1
fi

# Add all changes to the staging area
git add .

# Commit changes with the provided message
git commit -m "$1"

# Push changes to the 'main' branch
git push -u origin main