import cv2
import os

# Directory containing the image frames
frames_directory = '/Users/joseochoa/Desktop/Stereogram Project/output_stereograms'
output_video_path = 'output_video.mp4'

# Get a list of all image filenames in the directory
frame_files = sorted([f for f in os.listdir(frames_directory) if f.endswith('.png')])

# Read the first frame to get image dimensions
sample_frame = cv2.imread(os.path.join(frames_directory, frame_files[0]))
height, width, layers = sample_frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

for frame_file in frame_files:
    # Read the frame image
    frame = cv2.imread(os.path.join(frames_directory, frame_file))

    # Write the frame to the video
    out.write(frame)

# Release the VideoWriter object
out.release()

print(f"Video saved as {output_video_path}")
