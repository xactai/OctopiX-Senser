import cv2
from yolov8 import YOLOv8
from save_video import SaveVideo
import argparse
import numpy as np
import os
import socket
import subprocess


ROOT_PATH = os.path.join(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'
video_extn = ["mp4", "mkv", "xvi"]
image_extn = ["jpg", "jpeg", "png"]

parser = argparse.ArgumentParser()


parser.add_argument("--model", type=str, required=False, choices=['yolov8', 'yolov11', 'yolo11s'], default="yolov8")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--image_path", type=str, required=False, default="chairs.jpg")

args = parser.parse_args()
backend = args.backend


if backend == "cpu":
    model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.{MODEL_EXTENSION}' if args.model in ['yolov8', 'yolov11'] else f'{args.model}.{MODEL_EXTENSION}')
elif backend == 'npu':
    model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.qdq.{MODEL_EXTENSION}' if args.model in ['yolov8', 'yolov11'] else f'{args.model}.qdq.{MODEL_EXTENSION}')
elif backend == "cpuqdq":
    print("Inferencing Quantized model on CPU")
    model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.qdq.{MODEL_EXTENSION}' if args.model in ['yolov8', 'yolov11'] else f'{args.model}.qdq.{MODEL_EXTENSION}')
    backend = "cpu"

yolov8_detector = YOLOv8(model_path, conf_thres=0.04, iou_thres=0.3, backend=backend)


if args.image_path != 'camera':
    if args.image_path.endswith(tuple(video_extn)):
        filename = os.path.basename(args.image_path).split('.')[0]
        fileextn = os.path.basename(args.image_path).split('.')[-1]
        model_name = os.path.basename(args.model).split('.')[0]

        cap = cv2.VideoCapture(os.path.join(ROOT_PATH, 'assets', 'images', f'{args.image_path}'))
        if (cap.isOpened()):
            pass
        else:
            print("File not opened.")
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_saver = SaveVideo(image_size=[width, height],
                                video_path=f"C:/volterra_research/old/results/{filename}_{model_name}_{backend}_output.{fileextn}")
        while True:  
            res, image = cap.read()
            if res:
                # Detect Objects
                boxes, scores, class_ids, _ = yolov8_detector(image)

                # Draw detections
                output_image = yolov8_detector.draw_detections(image)
                cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
                cv2.imshow('Detected Objects',output_image)
                video_saver.save(output_image)
                key = cv2.waitKey(1)
                if key==0x27 or key==ord('q'):
                    break
            else:
                break
        video_saver.release()
        cap.release()

    elif args.image_path.endswith(tuple(image_extn)):
        filename = os.path.basename(args.image_path).split('.')[0]
        fileextn = os.path.basename(args.image_path).split('.')[-1]
        model_name = os.path.basename(args.model).split('.')[0]
        image_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Detect Objects
        boxes, scores, class_ids, _ = yolov8_detector(img)

        # Draw detections
        output_image = yolov8_detector.draw_detections(img)
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Objects", output_image)
        cv2.waitKey(0)

    else:
        buffer_size = 2*1024*1024
        try:
            print("Loading RTSP")
            subprocess.Popen([r"C:\venvs\python3.11.9.amd64\Scripts\python.exe", os.path.join(os.path.dirname(os.path.dirname(__file__)), "_rtsp_", "client.py")])
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(("localhost", 8080))
            server.listen(1)

            conn, addr = server.accept()

            data = b""
            cv2.namedWindow('Detected Objects',cv2.WINDOW_NORMAL)

            while True:
                packet = conn.recv(buffer_size)
                if not packet:
                    print("Exiting")
                    exit(0)

                if packet == b"end":
                    received_data = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(received_data, cv2.IMREAD_COLOR)

                    # Detect Objects
                    boxes, scores, class_ids, _ = yolov8_detector(image)
                    # Draw detections
                    output_image = yolov8_detector.draw_detections(image)
                    cv2.imshow('Detected Objects',output_image)
                    key = cv2.waitKey(1)
                    data = b""
                    if key==0x27 or key==ord('q'):
                        conn.close()
                        exit(0)
                else: data += packet

        except Exception as e:
            print(f"Error occurred: {e}")

# for webcam
elif args.image_path == "camera":
    cap = cv2.VideoCapture(0)
    while True:  
        res, image = cap.read()
        if res:
            # Detect Objects
            boxes, scores, class_ids, _ = yolov8_detector(image)

            # Draw detections
            output_image = yolov8_detector.draw_detections(image)
            cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
            cv2.imshow('Detected Objects',output_image)
            key = cv2.waitKey(1)
            if key==0x27 or key==ord('q'):
                break
        else:
            break
    cap.release()

    