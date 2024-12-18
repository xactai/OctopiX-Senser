import cv2
from yoloseg.YoloSeg import YOLOSeg
from ffnetseg.ffnethub import HubFFnetSeg
import argparse
import os

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type=str, required=False, default="chairs.jpg")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--model", required=False, type=str, choices=['yolo','ffnet78'])

args = parser.parse_args()

filename = args.image_path.split(".")[0]

# Read image from HDD
image_path = os.path.join(ROOT_PATH, "assets", "images", args.image_path)
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

backend = args.backend

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
    # Detect Objects
    boxes, scores, class_ids, masks = yoloseg(img)
    # Draw detections
    combined_img = yoloseg.draw_masks(img)

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
    combined_img = ffnetseg(img)


print(f'using model: {os.path.basename(model_path)}, using backend: {backend}')

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite(f"./assets/output/{filename}_{backend}_{args.model}.jpg", combined_img)
cv2.waitKey(0)