#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: No commit message provided."
    echo "Usage: ./push.sh \"Your commit message\" [-f|--force]"
    exit 1
fi

# Check for the force flag (-f or --force)
FORCE_PUSH=false
if [ "$2" == "-f" ] || [ "$2" == "--force" ]; then
    FORCE_PUSH=true
fi

# Add all changes to the staging area
git add .

# Commit changes with the provided message
git commit -m "$1"

# Push changes to the 'main' branch
if [ "$FORCE_PUSH" == true ]; then
    echo "Performing a force push..."
    git push -u origin main --force
else
    git push -u origin main
fi
