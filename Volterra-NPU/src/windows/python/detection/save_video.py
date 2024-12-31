import cv2
import numpy as np


class SaveVideo():
    def __init__(self, image_size=[640, 480], video_path="output.avi", fps=30):
        self.image_size = image_size
        self.video_path = video_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter(self.video_path, self.fourcc, self.fps, (np.int64(self.image_size[0]), np.int64(self.image_size[1])))

    def save(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()


if __name__ == "__main__":
    pass
