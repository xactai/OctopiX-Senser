import cv2 as cv
import numpy as np
import time
import socket
import os

time.sleep(2)

client_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_soc.connect(("localhost", 8080))

rtsp_url = "rtsp://admin:123456@192.168.10.220:554/mode=real&idc=3&ids=2"

def send_data(data):
    val = client_soc.send(data)
    # print("val: ", val)
    val = client_soc.send(b"end")


cap = cv.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("RTSP not opened!")
    exit(0)
else:
    ret, frame = cap.read()
    while ret:
        start = time.time()
        ret, frame = cap.read()
        _, encoded_data = cv.imencode(".png", frame)
        data = encoded_data.tobytes()
        send_data(data)
        end = time.time()
        # print(f"Frame time: {end - start:.4f}", end='')
    cap.release()
    client_soc.close()


