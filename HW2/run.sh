#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Give docker permission
sudo chmod 666 /var/run/docker.sock

# Create a Docker volume for data persistence
echo "Creating Docker volume: homework2-heart-disease"
docker volume create --name hw2-database

# Create a Docker network for container communication
echo "Creating Docker network: etl-database"
docker network create hw2-database


# Build Jupyter Docker image
echo "Building Jupyter Docker image from dockerfile-HW2"
docker build -f dockerfiles/dockerfile-HW2 -t jupyter-hw2-image .

# Run Jupyter container with volume and network setup
echo "Starting Jupyter container"
docker run -it --network hw2-database \
           --name hw2-container \
           -v ./src:/app/src \
           -v ./staging_data:/app/staging_data \
           -p 8888:8888 \
           jupyter-hw2-image
