import os
import cv2
import numpy as np
import argparse
from classification.classification import Classification
from classification import utils
import socket
import subprocess


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_EXTENSION = 'onnx'

image_extn = ["jpg", "png", "jpeg"]
video_extn = ["mp4", "avi", "mkv"]
output_window = "Output"

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
cv2.namedWindow(output_window, cv2.WINDOW_NORMAL)

if args.image_path.endswith(tuple(image_extn)):
    image_path = os.path.join(ROOT_PATH, 'assets', 'images', f'{args.image_path}')
    image = cv2.imread(image_path)
    for _ in range(100):
        class_name, confidence_value = cfcn.process_image(image)
    output_image = utils.display_result(image, class_name, confidence_value)
    print(class_name, confidence_value)
    cv2.imshow(output_window, output_image)
    cv2.waitKey(0)
    exit(0)
elif args.image_path.endswith(tuple(video_extn)):
    cap = cv2.VideoCapture(os.path.join(ROOT_PATH, 'assets', 'images', f'{args.image_path}'))
    ret = True
    while ret:
        ret, frame = cap.read()
        class_name, confidence_value = cfcn.process_image(frame)
        output_image = utils.display_result(frame, class_name, confidence_value)
        cv2.imshow(output_window, output_image)
        key = cv2.waitKey(1)
        if key==0x27 or key==ord('q'):
            exit(0)

elif args.image_path == 'rtsp':
    buffer_size = 2*1024*1024
    try:
        print("Loading RTSP")
        subprocess.Popen([r"C:\venvs\python3.11.9.amd64\Scripts\python.exe", os.path.join(os.path.dirname(os.path.dirname(__file__)), "_rtsp_", "client.py")])
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("localhost", 8080))
        server.listen(1)

        conn, addr = server.accept()

        data = b""

        while True:
            packet = conn.recv(buffer_size)
            if not packet:
                print("Exiting")
                exit(0)

            if packet == b"end":
                received_data = np.frombuffer(data, np.uint8)
                image = cv2.imdecode(received_data, cv2.IMREAD_COLOR)

                class_name, confidence_value = cfcn.process_image(image)
                output_image = utils.display_result(image, class_name, confidence_value)
                cv2.imshow(output_window, output_image)
                key = cv2.waitKey(1)
                data = b""
                if key==0x27 or key==ord('q'):
                    conn.close()
                    exit(0)

            else: data += packet
    except Exception as e:
        print(f"Error occured: {e}")

elif args.image_path == "camera":
    cap = cv2.VideoCapture(0)

while True:  
    res, image = cap.read()
    try:
        if res:
            class_name, confidence_value = cfcn.process_image(image)
            output_image = utils.display_result(image, class_name, confidence_value)
            cv2.imshow(output_window, output_image)
            key = cv2.waitKey(1)
            if key==0x27 or key==ord('q'):
                break
    except Exception as e:
        print(f"Error occured: {e}")









