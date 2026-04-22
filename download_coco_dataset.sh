#!/bin/bash

set -e  # Exit immediately if a command fails

# Create directory
mkdir -p datasets/coco_images

# Download COCO validation images
curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip

# Unzip into target directory
unzip -q val2017.zip -d datasets/coco_images

# Remove the zip file after extraction
rm val2017.zip

# Download COCO train/val 2017 annotations
curl -L -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip annotations into target directory
# Note: The zip file inherently contains an 'annotations' directory at its root.
# Extracting it to datasets/coco_images will result in datasets/coco_images/annotations/instances_val2017.json
unzip -q annotations_trainval2017.zip -d datasets/coco_images

# Remove the annotations zip file
rm annotations_trainval2017.zip

echo "COCO val2017 dataset and annotations downloaded and extracted successfully."