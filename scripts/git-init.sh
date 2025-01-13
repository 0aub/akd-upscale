#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found. Please create it with GITHUB_USERNAME, GITHUB_EMAIL, GITHUB_REPO_NAME, and GITHUB_SSH_KEY_HASH."
    exit 1
fi

# Step 1: Configure Git
echo "Configuring Git with your username and email..."
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

# Step 2: Check for SSH key and validate fingerprint
echo "Checking for SSH key and validating fingerprint..."
SSH_KEY_PATH=~/.ssh/id_ed25519
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "SSH key not found, generating a new SSH key..."
    ssh-keygen -t ed25519 -C "$GITHUB_EMAIL" -f "$SSH_KEY_PATH" -N ""
    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY_PATH"
    echo "SSH key generated. Please add the following public key to your GitHub account:"
    cat "${SSH_KEY_PATH}.pub"
    echo "Visit https://github.com/settings/keys to add your SSH key."
    read -p "Press Enter after you have added your SSH key to GitHub."
else
    echo "SSH key exists. Using existing key."
    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY_PATH"
fi

# Validate SSH key fingerprint
CURRENT_SSH_KEY_HASH=$(ssh-keygen -lf "$SSH_KEY_PATH.pub" | awk '{print $2}')
if [ "$CURRENT_SSH_KEY_HASH" != "$GITHUB_SSH_KEY_HASH" ]; then
    echo "Error: SSH key fingerprint does not match the expected hash in .env."
    exit 1
fi

# Step 3: Initialize the Git repository
echo "Initializing Git repository..."
git init
git add .
git commit -m "Initial commit."

# Step 4: Adding remote GitHub repository
REMOTE_URL="git@github.com:$GITHUB_USERNAME/$GITHUB_REPO_NAME.git"
git remote add origin "$REMOTE_URL"
echo "Remote GitHub repository added."

# Step 5: Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main
echo "Repository setup complete! Your code has been pushed to: https://github.com/$GITHUB_USERNAME/$GITHUB_REPO_NAME"