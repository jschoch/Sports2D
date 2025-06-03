import cv2
import numpy as np
import onnxruntime
import os
from tqdm import tqdm

def generate_calibration_dataset(video_path, output_dir, num_frames=100, image_size=(416, 416)):  # Changed default resize
    """
    Generates a calibration dataset from a video file and saves it as .npy files.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the output directory for saving .npy files.
        num_frames (int): Number of frames to extract for calibration.  Defaults to 100.
        resize (tuple): Resize dimensions (width, height). Defaults to (416, 416).
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count < num_frames:
            resized_frame = cv2.resize(frame, image_size)
            # Convert to NumPy array and transpose dimensions for (B, C, H, W) format
            reshaped_frame = np.transpose(resized_frame, (2, 0, 1))  # Transpose!
            # Add a batch dimension
            reshaped_frame = reshaped_frame[np.newaxis, ...] # Adds batch size of 1

            filename = os.path.join(output_dir, f"calibration_frame_{frame_count}.npy")
            np.save(filename, reshaped_frame)  # Save the frame as a numpy array

            frame_count += 1
        else:
            break # Stop after extracting num_frames frames

    cap.release()
    print(f"Generated calibration dataset with {num_frames} frames in '{output_dir}'.")

if __name__ == "__main__":
    # Example Usage:  Replace these paths with your actual file locations!
    onnx_model_path = "/home/schoch/.cache/rtmlib/hub/checkpoints/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"  # Path to your YOLO-X ONNX model
    video_file_path = "/mnt/c/Files/new_swings/20241117-153031-left.mp4" # Path to your input video file
    calibration_output_dir = "calibration_images"
    image_size = (416,416)
    #create_calibration_dataset(onnx_model_path, video_file_path, calibration_output_dir, num_frames=100, image_size=image_size)
    generate_calibration_dataset( video_file_path, calibration_output_dir, num_frames=100, image_size=image_size)

