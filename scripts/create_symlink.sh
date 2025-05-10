#!/bin/bash

# Create the target directory if it doesn't exist
mkdir -p ~/workspace/reddit-tracker/docs

# Create the symbolic link
ln -sf ~/workspace/reddit-post-classifier/API_DOC.md ~/workspace/reddit-tracker/docs/classifier-ai.md

echo "Symbolic link created successfully!"