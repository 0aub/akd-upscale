#!/bin/bash

# Source the .env file to load environment variables
source .env
echo "Current working directory: $(pwd)"

# Check if GITHUB_REPO_NAME is set in the .env file
if [ -z "$GITHUB_REPO_NAME" ]; then
  echo "Error: GITHUB_REPO_NAME is not set in the .env file."
  exit 1
fi

# Variables
IMAGE_NAME="${GITHUB_REPO_NAME}_docker_image"               # Name of the Docker image
CONTAINER_NAME="${GITHUB_REPO_NAME}_docker_container"       # Name of the Docker container
FINAL_IMAGE_NAME="${GITHUB_REPO_NAME}_docker_image_final"   # Name of the final saved image

# Path to mount (change this to your project directory)
HOST_DIR=$(pwd)         # Current directory on host
CONTAINER_DIR="/app"    # Directory in container (as defined in Dockerfile)

# Step 1: Check if container already exists and remove it
if [ $(docker ps -aq -f name=^/$CONTAINER_NAME$) ]; then
    docker rm -f $CONTAINER_NAME || error_exit "Failed to remove existing container."
fi

# Step 2: Build the Docker image from the Dockerfile
echo "Building Docker image..."
docker build -t $IMAGE_NAME . || error_exit "Failed to build Docker image."

# Step 3: Run the container with volume mounting for real-time changes
echo "Running Docker container with volume mounting..."
docker run -it --name $CONTAINER_NAME --gpus all --privileged -v "$HOST_DIR:$CONTAINER_DIR" $IMAGE_NAME /bin/bash || error_exit "Failed to run Docker container."

# Step 4: Commit the container changes to a new image after you exit
echo "Committing container changes to a new image..."
docker commit $CONTAINER_NAME $FINAL_IMAGE_NAME || error_exit "Failed to commit changes to new image."

# Step 5: Clean up the original container and intermediate image
echo "Cleaning up..."
docker rm $CONTAINER_NAME || error_exit "Failed to remove container."
docker rmi $IMAGE_NAME || error_exit "Failed to remove image."

echo "Process completed. The final image is saved as $FINAL_IMAGE_NAME."