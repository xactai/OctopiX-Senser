from super_resolution.resolution import SuperResolution
import cv2
from PIL import Image
import numpy as np
import argparse
import os


ROOT_PATH = os.path.join(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'

output_dir = os.path.join(ROOT_PATH, "outputs")

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=False, choices=['real_esrgan_x4plus', 'xlsr',"quicksrnetsmall"], default="yolov8")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--image_path", type=str, required=False, default="Doll.jpg")

args = parser.parse_args()

backend = args.backend

if args.model == 'real_esrgan_x4plus':
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
if args.model == 'xlsr':
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
if args.model == 'quicksrnetsmall':
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
        backend = "cpu"

filename = args.image_path.split('.')[0]

sr = SuperResolution(
    path=model_path,
    backend=backend,
    nchw=False if args.model == "xlsr-tf-onnx" else True,
    data_type='uint8' if args.model == "xlsr-hf-prev" else 'float'
)

image = cv2.imread(os.path.join(ROOT_PATH, "assets", "images", args.image_path), cv2.IMREAD_COLOR)

resized_image = cv2.cvtColor(cv2.resize(image, (128, 128)), cv2.COLOR_BGR2RGB)

output = sr(image)

output_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

image = Image.fromarray(np.zeros((512, 1024, 3)).astype(np.uint8))

image.paste(Image.fromarray(resized_image), (192, 192))
image.paste(Image.fromarray(output_image), (512, 0))

image.save(os.path.join(ROOT_PATH, "outputs", f"{filename}_{args.model}_{args.backend}.jpg"))
