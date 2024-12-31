from .utils import model_dir, image_dir
import onnxruntime
import cv2
import time
import numpy as np
import math
from colorist import BgColor
import random


class HubFFnetSeg:
    def __init__(self, path, num_masks=19, backend='npu'):
        self.num_masks = num_masks
        self.backend = backend

        # Initialize model
        self.initialize_model(path)

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(19)]

        self.colors = [np.array(self.colors[i]) for i in range(19)]

    def __call__(self, image):
        return self.segment_objects(image)
    
    def segment_objects(self, image):
        start = time.perf_counter()
        input_tensor = self.prepare_input(image)
        end = time.perf_counter()
        str1 = f"Prepare input time: {(end - start)*1000:.5f} msecs"

        # Perform inference on the image
        inference_start = time.perf_counter()
        outputs = self.inference(input_tensor)
        inference_end = time.perf_counter()
        str2 = f"{BgColor.GREEN}Inference time: {(inference_end - inference_start)*1000:.5f} msecs{BgColor.OFF}"

        # Post-Processing
        start = time.perf_counter()
        x = np.array(input_tensor[0].transpose(1, 2, 0)*255).astype(np.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        final = np.zeros_like(x).astype(np.int32)
        alpha = 0.5
        for i in range(self.num_masks):
            img_ = np.array(outputs[0][0][i]).clip(0, 1)[..., np.newaxis]*255
            img_ = np.concatenate([img_, img_, img_], axis=-1)
            img_ = cv2.resize(img_, input_tensor.shape[-2:][::-1])
            img_ = np.array(img_ > 10).astype(np.uint8)
            temp = x * img_
            mask = img_*self.colors[i]
            temp = temp.astype(mask.dtype)
            y = cv2.addWeighted(mask, alpha, temp, 1-alpha, 0.0)
            final += y

        temp = np.array(final < np.array([10, 10, 10])).astype(np.uint8)*x
        temp = np.array(temp).astype(final.dtype)
        final += temp
        final = np.array(final.clip(0, 255), dtype=np.uint8)
        end = time.perf_counter()
        str3 = f"Post Processing time: {(end - start)*1000:.5f} msecs"
        str4 = f"End-to-End Prediction time: {(end - inference_start)*1000:.5f} msecs"
        
        desc = "\r" + " ".join([str1, str2, str3, str4])
        print(desc)
        
        return final

    def initialize_model(self, path):
        options = onnxruntime.SessionOptions()

        # (Optional) Enable configuration that raises an exception if the model can't be
        # run entirely on the QNN HTP backend.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
        self.session = onnxruntime.InferenceSession(path,
                                                    sess_options=options,
                                                    providers=["QNNExecutionProvider"] if self.backend == "npu" else ["CPUExecutionProvider"],
                                                    provider_options=[{"backend_path": "QnnHtp.dll" if self.backend == "npu" else "Qnncpu.dll"}])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor
    
    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]