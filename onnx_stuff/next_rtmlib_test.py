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

from quark.onnx import get_library_path as vai_lib_path

if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
    device = 'rocm'
    providers = ['ROCMExecutionProvider']
elif 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    device = 'CUDA'
    providers = ['CUDAExecutionProvider']
else:
    device = 'CPU'
    providers = ['CPUExecutionProvider']

sess_options = onnxruntime.SessionOptions()
sess_options.register_custom_ops_library(vai_lib_path(device))

#wholebody = Wholebody(to_openpose=openpose_skeleton,
                      #mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      #mode='performance',
                      #backend=backend, device=device)
#keypoints, scores = wholebody(img)


#modes = ['balanced', 'performance', 'lightweight']
modes = ['lightweight']
det_frequency = 1  # Detection frequency

custom = partial(
             Custom,
             to_openpose=False,
             det_class='YOLOX',
             det='testopt_det.onnx',
             #det_input_size=(640, 640),
             det_input_size= (416,416),
             pose_class='RTMPose',
             pose='testopt_halpe26.onnx',
             pose_input_size=(192, 256),
             #pose_input_size=(416,416),
             backend=backend,
             device=device)

# Run tests for each mode
for mode in modes:
    pose_tracker = PoseTracker(
        custom,
        det_frequency=det_frequency,
        mode=mode,  # Testing different modes
        backend=backend,
        device=device,
        tracking=False,
        #det="testopt.onnx",
        to_openpose=False
    )

    # Define function for time measurement
    def run_pose_tracker():
        keypoints, scores = pose_tracker(img)

    print(f"Running 100 runs for mode: {mode}")

    execution_time = timeit.timeit(run_pose_tracker, number=100)  # Runs the function 100 times
    print(f"Average execution time for '{mode}': {execution_time / 100:.6f} seconds\n")

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

#img_show = np.zeros(img_show.shape, dtype=np.uint8)


#  draw and display

#img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)
#cv2.imshow('img', img_show)
#cv2.waitKey()
