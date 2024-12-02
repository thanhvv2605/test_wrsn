import cv2
import os
import re
from natsort import natsorted
import sys

# Append the script's parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Directory containing the images
image_directory = os.path.join('image', 'test_image')

# Output video file name
output_video = "network_states_video.mp4"

# Frame rate of the video
frame_rate = 10

# Get list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

if not image_files:
    print("Error: No PNG images found in the specified directory.")
    exit()

# Sort the images based on the numeric value in their filenames
try:
    image_files = natsorted(image_files, key=lambda x: int(re.search(r'\d+', x).group()))
except AttributeError as e:
    print(f"Error sorting images: {e}. Ensure filenames contain numeric values.")
    exit()

# Read the first image to get the dimensions
first_image_path = os.path.join(image_directory, image_files[0])
first_image = cv2.imread(first_image_path)

if first_image is None:
    print(f"Error: Unable to read the first image: {first_image_path}. Check file format or corruption.")
    exit()

height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Add each image to the video
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Warning: Unable to read image: {image_path}. Skipping this file.")
        continue

    video.write(frame)

# Release the video writer
video.release()

print(f"Video saved as {output_video}")