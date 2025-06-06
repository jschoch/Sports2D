import cv2
import numpy as np
import img_show
import timeit
import onnxruntime

from rtmlib import Wholebody,Custom, draw_skeleton, PoseTracker, BodyWithFeet
from functools import partial

device = 'rocm'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
img = cv2.imread('/mnt/c/Files/screenshots/test.jpg')



#modes = ['balanced', 'performance', 'lightweight']
modes = ['lightweight']
det_frequency = 10  # Detection frequency

custom = partial(
             Custom,
             to_openpose=False,
             det_class='YOLOX',
             #det='/home/schoch/.cache/rtmlib/hub/checkpoints/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx',
             det='testopt_det.onnx',
             #det='testopt.onnx',
             #det_input_size=(640, 640),
             det_input_size= (416,416),
             pose_class='RTMPose',
             #pose='testopt_halpe26.onnx',
             #pose='/home/schoch/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx',
             #pose = "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.onnx",
             pose = "RTMW_x.onnx",
             #pose_input_size=(192, 256),
             pose_input_size=(288,384),
             #pose_input_size=(416,416),
             backend=backend,
             device=device)


# Run tests for each mode
pose_tracker = PoseTracker(
        custom,
        det_frequency=det_frequency,
        mode='lightweight',  # Testing different modes
        backend=backend,
        device=device,
        tracking=False,
        to_openpose=False
    )
for mode in modes:
    # Define function for time measurement
    def run_pose_tracker():
        keypoints, scores = pose_tracker(img)

    runs = 1000
    print(f"Running {runs} runs for mode: {mode}")

    execution_time = timeit.timeit(run_pose_tracker, number=runs)  # Runs the function 100 times
    print(f"Average execution time for '{mode}': {execution_time / runs:.6f} seconds\n")

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

#img_show = np.zeros(img_show.shape, dtype=np.uint8)


#  draw and display

#img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)
#cv2.imshow('img', img_show)
#cv2.waitKey()
