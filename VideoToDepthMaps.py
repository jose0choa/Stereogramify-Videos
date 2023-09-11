import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Load the DPT depth estimation model and image processor
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Open the video file
video_path = '/Users/joseochoa/Desktop/Stereogram Project/Tony Ferguson knocked out by Mike Chandler   UFC 274   Great Knockout copy.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Define a function to preprocess and estimate depth for each frame
def process_frame(frame):
    image = Image.fromarray(frame)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth

frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Process the frame for depth estimation
    depth_image = process_frame(frame)

    # Save or display the depth image as needed
    depth_image.save(f'depth_frame_{frame_count:04d}.png')
    frame_count += 1

# Release the video capture object and close the video file
cap.release()

print(f"Processed {frame_count} frames for depth estimation")
