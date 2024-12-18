import os
import cv2
import argparse
from classification.classification import Classification
from classification import utils

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type=str, required=False, default="keyboard.jpg")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--model", required=False, choices=["inception", "yolo"], default="yolo")

args = parser.parse_args()

backend= args.backend

if args.model == 'inception':
    if backend == 'cpu':
        path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}_v3.{MODEL_EXTENSION}')
    elif backend == 'npu':
        path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}_v3.qdq.{MODEL_EXTENSION}')
    elif backend == 'cpuqdq':
        path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}_v3.qdq.{MODEL_EXTENSION}')
        backend = 'cpu'
elif args.model == 'yolo':
    if backend == 'cpu':
        path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}v8n-cls.{MODEL_EXTENSION}')
    if backend == 'npu':
        path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}v8n-cls.qdq.{MODEL_EXTENSION}')
    if backend == 'cpuqdq':
        path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}v8n-cls.qdq.{MODEL_EXTENSION}')
        backend = 'cpu'
else:
    raise f"{args.model} is not available"

cfcn = Classification(path, backend)
if args.image_path != 'camera':
    image_path = os.path.join(ROOT_PATH, 'assets', 'images', f'{args.image_path}')
    image = cv2.imread(image_path)
    class_name, confidence_value = cfcn.process_image(image)
    output_image = utils.display_result(image, class_name, confidence_value)
    print(class_name, confidence_value)
    cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
    cv2.imshow('Output',output_image)
    cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(0)
    while True:  
        res, image = cap.read()
        if res:
            class_name, confidence_value = cfcn.process_image(image)
            output_image = utils.display_result(image, class_name, confidence_value)
            print(class_name, confidence_value)
            cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
            cv2.imshow('Output',output_image)
            key = cv2.waitKey(1)
            if key==0x27 or key==ord('q'):
                break









