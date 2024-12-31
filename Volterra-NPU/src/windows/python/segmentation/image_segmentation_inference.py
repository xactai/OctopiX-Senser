import cv2
from yoloseg.YoloSeg import YOLOSeg
from ffnetseg.ffnethub import HubFFnetSeg
import argparse
import os
from ffnetseg import utils
import numpy as np
import subprocess
import socket


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'

video_extn = ["mkv", "mp4", "avi"]
image_extn = ["png", "jpg", "jpeg"]

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type=str, required=False, default="chairs.jpg")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--model", required=False, type=str, choices=['yolo','ffnet78','ffnet40'])


args = parser.parse_args()

backend = args.backend


def inference_yolo(img):
    # Detect Objects
    boxes, scores, class_ids, masks = yoloseg(img)
    # Draw detections
    combined_img = yoloseg.draw_masks(img)
    return combined_img

output_window = "Output"

cv2.namedWindow(output_window, cv2.WINDOW_NORMAL)

# Initialize YOLOv5 Instance Segmentator
if args.model == 'yolo':
    if backend == "npu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}v8n-seg.qdq.{MODEL_EXTENSION}')
    elif backend == 'cpu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}v8n-seg.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}v8n-seg.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
    yoloseg = YOLOSeg(model_path, conf_thres=0.4, iou_thres=0.3, backend=backend)

    # for image
    if args.image_path.endswith(tuple(image_extn)):
        # Read image from HDD
        image_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        combined_img = inference_yolo(img)
        cv2.imshow(output_window, combined_img)
        key = cv2.waitKey(0)
        if key == 'q' or key == 27:
            print("Exit")
            exit(0)
    elif args.image_path.endswith(tuple(video_extn)):
        cap = cv2.VideoCapture(os.path.join(ROOT_PATH, "assets", "images", args.image_path))
        if not cap.isOpened():
            print("File not opened!")
            exit(0)
        else:
            ret = True
            while ret:
                ret, frame = cap.read()
                combined_img = inference_yolo(frame)
                cv2.imshow(output_window, combined_img)
                key = cv2.waitKey(1)
                if key == 'q' or key == 27:
                    exit(0)
    elif args.image_path == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("File not opened!")
            exit(0)
        else:
            ret = True
            while ret:
                ret, frame = cap.read()
                combined_img = inference_yolo(frame)
                cv2.imshow(output_window, combined_img)
                key = cv2.waitKey(1)
                if key == 'q' or key == 27:
                    exit(0)
    elif args.image_path == "rtsp":
        buffer_size = 2*1024*1024
        try:
            print("Loading RTSP")
            subprocess.Popen([r"C:\venvs\python3.11.9.amd64\Scripts\python.exe", os.path.join(os.path.dirname(os.path.dirname(__file__)), "_rtsp_", "client.py")])
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("localhost", 8080))
            server.listen(1)

            conn, addr = server.accept()

            data = b""
            cv2.namedWindow(output_window,cv2.WINDOW_NORMAL)

            while True:
                packet = conn.recv(buffer_size)
                if not packet:
                    print("Exiting")
                    exit(0)

                if packet == b"end":
                    received_data = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(received_data, cv2.IMREAD_COLOR)
                    combined_img = inference_yolo(image)
                    cv2.imshow(output_window, combined_img)
                    key = cv2.waitKey(1)
                    data = b""
                    if key == 'q' or key == 27:
                        exit(0)
                else: data += packet

        except Exception as e:
            print(f"Error occurred: {e}")


elif args.model == 'ffnet78':
    if backend == "npu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}s.qdq.{MODEL_EXTENSION}')
    elif backend == 'cpu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}s.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}s.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
    ffnetseg = HubFFnetSeg(model_path, backend=backend)
    # for image
    if args.image_path.endswith(tuple(image_extn)):
        # Read image from HDD
        image_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        combined_img = ffnetseg(img)
        cv2.imshow(output_window, combined_img)
        key = cv2.waitKey(0)
        if key == 'q' or key == 27:
            print("Exit")
            exit(0)
    elif args.image_path.endswith(tuple(video_extn)):
        cap = cv2.VideoCapture(os.path.join(ROOT_PATH, "assets", "images", args.image_path))
        if not cap.isOpened():
            print("File not opened!")
            exit(0)
        else:
            ret = True
            while ret:
                ret, frame = cap.read()
                combined_img = ffnetseg(frame)
                cv2.imshow(output_window, combined_img)
                key = cv2.waitKey(1)
                if key == 'q' or key == 27:
                    exit(0)
    elif args.image_path == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("File not opened!")
            exit(0)
        else:
            ret = True
            while ret:
                ret, frame = cap.read()
                combined_img = ffnetseg(frame)
                cv2.imshow(output_window, combined_img)
                key = cv2.waitKey(1)
                if key == 'q' or key == 27:
                    exit(0)
    elif args.image_path == "rtsp":
        buffer_size = 2*1024*1024
        try:
            print("Loading RTSP")
            subprocess.Popen([r"C:\venvs\python3.11.9.amd64\Scripts\python.exe", r"C:\volterra_research\refactored_code\rtsp\client.py"])
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("localhost", 8080))
            server.listen(1)

            conn, addr = server.accept()

            data = b""
            cv2.namedWindow(output_window,cv2.WINDOW_NORMAL)

            while True:
                packet = conn.recv(buffer_size)
                if not packet:
                    print("Exiting")
                    exit(0)

                if packet == b"end":
                    received_data = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(received_data, cv2.IMREAD_COLOR)
                    combined_img = ffnetseg(image)
                    cv2.imshow(output_window, combined_img)
                    key = cv2.waitKey(1)
                    data = b""
                    if key == 'q' or key == 27:
                        exit(0)
                else: data += packet

        except Exception as e:
            print(f"Error occurred: {e}")

elif args.model == 'ffnet40':
    if backend == "npu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}s.qdq.{MODEL_EXTENSION}')
    elif backend == 'cpu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}s.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}s.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
    ffnetseg = HubFFnetSeg(model_path, backend=backend)
    # for image
    if args.image_path.endswith(tuple(image_extn)):
        # Read image from HDD
        image_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        combined_img = ffnetseg(img)
        cv2.imshow(output_window, combined_img)
        key = cv2.waitKey(0)
        if key == 'q' or key == 27:
            print("Exit")
            exit(0)
    elif args.image_path.endswith(tuple(video_extn)):
        cap = cv2.VideoCapture(os.path.join(ROOT_PATH, "assets", "images", args.image_path))
        if not cap.isOpened():
            print("File not opened!")
            exit(0)
        else:
            ret = True
            while ret:
                ret, frame = cap.read()
                combined_img = ffnetseg(frame)
                cv2.imshow(output_window, combined_img)
                key = cv2.waitKey(1)
                if key == 'q' or key == 27:
                    exit(0)
    elif args.image_path == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("File not opened!")
            exit(0)
        else:
            ret = True
            while ret:
                ret, frame = cap.read()
                combined_img = ffnetseg(frame)
                cv2.imshow(output_window, combined_img)
                key = cv2.waitKey(1)
                if key == 'q' or key == 27:
                    exit(0)
    elif args.image_path == "rtsp":
        buffer_size = 2*1024*1024
        try:
            print("Loading RTSP")
            subprocess.Popen([r"C:\venvs\python3.11.9.amd64\Scripts\python.exe", r"C:\volterra_research\refactored_code\rtsp\client.py"])
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("localhost", 8080))
            server.listen(1)

            conn, addr = server.accept()

            data = b""
            cv2.namedWindow(output_window,cv2.WINDOW_NORMAL)

            while True:
                packet = conn.recv(buffer_size)
                if not packet:
                    print("Exiting")
                    exit(0)

                if packet == b"end":
                    received_data = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(received_data, cv2.IMREAD_COLOR)
                    combined_img = ffnetseg(image)
                    cv2.imshow(output_window, combined_img)
                    key = cv2.waitKey(1)
                    data = b""
                    if key == 'q' or key == 27:
                        exit(0)
                else: data += packet

        except Exception as e:
            print(f"Error occurred: {e}")