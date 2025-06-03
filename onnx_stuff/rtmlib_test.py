import cv2
import numpy as np
import img_show
import timeit

from rtmlib import Wholebody, draw_skeleton, PoseTracker, BodyWithFeet

device = 'rocm'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
img = cv2.imread('/mnt/c/Files/screenshots/test.jpg')


#wholebody = Wholebody(to_openpose=openpose_skeleton,
                      #mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      #mode='performance',
                      #backend=backend, device=device)
#keypoints, scores = wholebody(img)


#modes = ['balanced', 'performance', 'lightweight']
modes = ['lightweight']
det_frequency = 1  # Detection frequency

# Run tests for each mode
for mode in modes:
    pose_tracker = PoseTracker(
        BodyWithFeet,
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



#  draw and display

#img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)
#cv2.imshow('img', img_show)
#cv2.waitKey()
