#!/bin/bash

# This script runs ViTPose and BEV on your images and then starts the optimization with BUDDI.
# If you have OpenPose installed, you can also run OpenPose on your images as well 
# and set the datasets.demo.openpose_folder to the folder where the OpenPose keypoints are stored.

# Usage: ./custom_demo.sh <input_image_path> <output_directory> <gpu_number>
# Example: ./custom_demo.sh /path/to/image.jpg /path/to/output 0

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_image_path> <output_directory> <gpu_number>"
    echo "Example: $0 /path/to/image.jpg /path/to/output 0"
    exit 1
fi

INPUT_IMAGE="$1"
OUTPUT_DIR="$2"
GPU_NUMBER="$3"

# Validate input image exists
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: Input image '$INPUT_IMAGE' does not exist or is not a file."
    exit 1
fi

# Validate GPU number is a valid integer
if ! [[ "$GPU_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU number '$GPU_NUMBER' must be a non-negative integer."
    exit 1
fi

# Set GPU device
echo "Using GPU device: $GPU_NUMBER"
export CUDA_VISIBLE_DEVICES="$GPU_NUMBER"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set up intermediate directories within output directory
IMAGES_DIR="$OUTPUT_DIR/images_OCHuman"
VITPOSE_DIR="$OUTPUT_DIR/vitpose_OCHuman"
BEV_DIR="$OUTPUT_DIR/bev_OCHuman"
OPTIMIZATION_DIR="$OUTPUT_DIR/optimization"

# Create intermediate directories
mkdir -p "$IMAGES_DIR"
mkdir -p "$VITPOSE_DIR"
mkdir -p "$BEV_DIR"
mkdir -p "$OPTIMIZATION_DIR"

# Copy input image to images directory
echo "Copying input image to working directory..."
cp "$INPUT_IMAGE" "$IMAGES_DIR/"

# Function to cleanup intermediate results
cleanup() {
    echo "Cleaning up intermediate results..."
    rm -rf "$IMAGES_DIR"
    rm -rf "$VITPOSE_DIR"
    rm -rf "$BEV_DIR"
    echo "Intermediate results cleaned up. Final optimization results preserved in: $OPTIMIZATION_DIR"
}

# Set trap to cleanup on exit (success or failure)
trap cleanup EXIT

# Run ViTPose on your image
echo "Running ViTPose on your image"

python llib/utils/keypoints/vitpose_model.py --image_folder "$IMAGES_DIR" --out_folder "$VITPOSE_DIR"



# Run BEV on your images 
echo "Running BEV on your images"

for image in "$IMAGES_DIR"/*; do
    # get only image name
    image_name=$(basename "$image")
    bev -i "$image" -o "$BEV_DIR/$image_name"
done


# Run OpenPose on your images 
echo "Not Running OpenPose on your images"
# We don't have OpenPose installed in this repo and run the demo on random images
# with keypoints detected by ViTPose (wholebody) model only. This more recent model 
# works quite well and is a good alternative to OpenPose.

# If you wish to run the original version with ViTPose (core body) + OpenPose (wholebody) keypoints, 
# you can install OpenPose from here https://github.com/CMU-Perceptual-Computing-Lab/openpose
# and use this command: ./build/examples/openpose/openpose.bin --image_dir demo/data/images_OCHuman --face --hand
# I keep it in openpose/examples/tutorial_api_python/ folder
# Here is a script I use for OpenPose: llib/utils/keypoints/run_openpose_folder.py
# OpenPose Docs: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md

# To pass OpenPose keypoints to the optimization script set
# datasets.demo.openpose_folder to the folder where the OpenPose keypoints are stored.

# Run Optimization with BUDDI
echo "Running Optimization with BUDDI"

# Run Optimization with BUDDI
# For OCHuman demo we set the openpose_folder to none
# If you have OpenPose installed, you can pass the openpose keypoints folder to the openpose_folder
# to test without demo images set datasets.demo.openpose_folder=keypoints/keypoints
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml --exp-opts logging.base_folder="$OPTIMIZATION_DIR/buddi_cond_bev_demo_OCHuman" datasets.train_names=['demo'] datasets.train_composition=[1.0] datasets.demo.original_data_folder="$OUTPUT_DIR" datasets.demo.image_folder=images_OCHuman datasets.demo.bev_folder=bev_OCHuman datasets.demo.vitpose_folder=vitpose_OCHuman datasets.demo.openpose_folder=none model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml logging.run=fit_buddi_cond_bev_flickrci3ds

echo "BUDDI optimization completed. Results saved to: $OPTIMIZATION_DIR"
echo "Final results are preserved in the output directory: $OUTPUT_DIR"
