import cv2

from yolov8 import YOLOv8
from save_video import SaveVideo
import argparse
import os

ROOT_PATH = os.path.join(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'
video_extn = ["mp4", "mkv", "xvi"]

parser = argparse.ArgumentParser()


parser.add_argument("--model", type=str, required=False, choices=['yolov8', 'yolov11'], default="yolov8")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--image_path", type=str, required=False, default="chairs.jpg")

args = parser.parse_args()
backend = args.backend


if args.model == 'yolov8':
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3, backend=backend)
if args.model == "yolov11":
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}n.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3, backend=backend)


if args.image_path != 'camera':
    if args.image_path.endswith(tuple(video_extn)):
        filename = os.path.basename(args.image_path).split('.')[0]
        fileextn = os.path.basename(args.image_path).split('.')[-1]
        model_name = os.path.basename(args.model).split('.')[0]
        video_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)

        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened()):
            pass
        else:
            print("File not opened.")
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_saver = SaveVideo(image_size=[width, height],
                                video_path=f"./assets/output/{filename}_{model_name}_{backend}_output.{fileextn}")
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
    else:
        image_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Detect Objects
        boxes, scores, class_ids, _ = yolov8_detector(img)

        # Draw detections
        output_image = yolov8_detector.draw_detections(img)
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Objects", output_image)
        cv2.waitKey(0)

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
    