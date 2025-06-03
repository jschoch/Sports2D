import cv2
import numpy as np
import onnxruntime
import os
from tqdm import tqdm

def create_calibration_dataset(onnx_file, video_path, output_dir, num_frames=100, image_size=(640, 640)):
    """
    Creates a calibration dataset from a video file for Quark quantization of YOLO-X.

    Args:
        onnx_file (str): Path to the YOLO-X ONNX model file.
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the calibration images.
        num_frames (int, optional): Number of frames to extract for calibration. Defaults to 100.
        image_size (tuple, optional): Desired size of the calibration images (width, height). Defaults to (640, 640).
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: Frames={frame_count}, FPS={fps}")

    # Calculate frame interval for sampling
    if num_frames > frame_count:
        num_frames = frame_count  # Use all frames if requested more than available
    frame_interval = max(1, int(frame_count / num_frames)) # Ensure at least 1 frame is sampled

    # Load ONNX model to get input name and shape. This helps ensure correct preprocessing.
    session = onnxruntime.InferenceSession(onnx_file)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Check if the image size matches the expected input shape of the model
    if tuple(input_shape[2:]) != image_size:
        print(f"Warning: Image size {image_size} does not match the ONNX model's input shape {tuple(input_shape[2:])}.  Resizing may affect accuracy.")

    # Extract frames and save as calibration images
    frame_num = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            resized_frame = cv2.resize(frame, image_size)  # Resize to the desired size
            #image_path = os.path.join(output_dir, f"calibration_{frame_num}.jpg")
            #cv2.imwrite(image_path, resized_frame)
            filename = os.path.join(output_dir, f"calibration_frame_{frame_count}.npy")
            np.save(filename, resized_frame)  # Save the frame as a numpy array

        frame_num += 1

    cap.release()
    print(f"Calibration dataset created in {output_dir} with {len(os.listdir(output_dir))} images.")


if __name__ == "__main__":
    # Example Usage:  Replace these paths with your actual file locations!
    onnx_model_path = "/home/schoch/.cache/rtmlib/hub/checkpoints/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"  # Path to your YOLO-X ONNX model
    video_file_path = "/mnt/c/Files/new_swings/20241117-153031-left.mp4" # Path to your input video file
    calibration_output_dir = "calibration_images"
    image_size = (416,416)
    create_calibration_dataset(onnx_model_path, video_file_path, calibration_output_dir, num_frames=100, image_size=image_size)

