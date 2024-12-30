from super_resolution.resolution import SuperResolution
import cv2
from PIL import Image
import numpy as np
import argparse
import os
import socket
import subprocess


ROOT_PATH = os.path.join(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'

output_dir = os.path.join(ROOT_PATH, "outputs")

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=False, choices=['real_esrgan_x4plus', 'xlsr', 'xlsr-hf', 'xlsr-tf-onnx', 'xlsr-hf-prev', "quicksrnetsmall"], default="yolov8")
parser.add_argument("--backend", type=str, required=False, default="cpu")
parser.add_argument("--image_path", type=str, required=False, default="Doll.jpg")

args = parser.parse_args()

backend = args.backend

video_extn = ["mkv", "mp4", "avi"]
image_extn = ["png", "jpg", "jpeg"]


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
if args.model == 'xlsr-hf':
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
if args.model == 'xlsr-hf-prev':
    if backend == "cpu":
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.{MODEL_EXTENSION}')
    elif backend == 'npu':
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
    elif backend == "cpuqdq":
        print("Inferencing Quantized model on CPU")
        model_path = os.path.join(ROOT_PATH, 'assets', 'models', f'{args.model}.qdq.{MODEL_EXTENSION}')
        backend = "cpu"
if args.model == 'xlsr-tf-onnx':
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

if args.image_path.endswith(tuple(image_extn)):
    # for image
    image = cv2.imread(os.path.join(ROOT_PATH, "assets", "images", args.image_path), cv2.IMREAD_COLOR)
    resized_image = cv2.cvtColor(cv2.resize(image, (128, 128)), cv2.COLOR_BGR2RGB)
    output = sr(image)
    output_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np.zeros((512, 1024, 3)).astype(np.uint8))
    image.paste(Image.fromarray(resized_image), (192, 192))
    image.paste(Image.fromarray(output_image), (512, 0))
    image.save(os.path.join(ROOT_PATH, "outputs", f"{filename}_{args.model}_{args.backend}.jpg"))
    opencv_image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("Output", opencv_image)
    key = cv2.waitKey(0)
    if key == 'q' or key == 27:
        exit(0)

elif args.image_path.endswith(tuple(video_extn)):
    # for video
    cap = cv2.VideoCapture(os.path.join(ROOT_PATH, "assets", "images", args.image_path))
    if not cap.isOpened():
        print("File not opened!")
        exit(0)
    else:
        ret = True
        while ret:
            ret, frame = cap.read()
            resized_image = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
            output = sr(resized_image)
            output_image = output
            image = Image.fromarray(np.zeros((512, 1024, 3)).astype(np.uint8))
            image.paste(Image.fromarray(resized_image), (192, 192))
            image.paste(Image.fromarray(output_image), (512, 0))
            opencv_image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow("Output", opencv_image)
            key = cv2.waitKey(1)
            if key == 'q' or key == 27:
                exit(0)

elif args.image_path == "camera":
    # for camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("File not opened!")
        exit(0)
    else:
        ret = True
        while ret:
            ret, frame = cap.read()
            resized_image = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
            output = sr(resized_image)
            output_image = output
            image = Image.fromarray(np.zeros((512, 1024, 3)).astype(np.uint8))
            image.paste(Image.fromarray(resized_image), (192, 192))
            image.paste(Image.fromarray(output_image), (512, 0))
            opencv_image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow("Output", opencv_image)
            key = cv2.waitKey(1)
            if key == 'q' or key == 27:
                exit(0)

elif args.image_path == "rtsp":
    # for rtsp
    output_window = "Output"
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

                resized_image = cv2.cvtColor(cv2.resize(image, (128, 128)), cv2.COLOR_BGR2RGB)
                output = sr(resized_image)
                output_image = output
                image = Image.fromarray(np.zeros((512, 1024, 3)).astype(np.uint8))
                image.paste(Image.fromarray(resized_image), (192, 192))
                image.paste(Image.fromarray(output_image), (512, 0))
                opencv_image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow(output_window, opencv_image)
                key = cv2.waitKey(1)
                data = b""
                if key == 'q' or key == 27:
                    exit(0)
            else: data += packet

    except Exception as e:
        print(f"Error occurred: {e}")

