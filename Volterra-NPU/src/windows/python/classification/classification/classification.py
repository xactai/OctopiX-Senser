import os
import onnxruntime
import time
import cv2
import numpy as np

from classification import utils

class Classification():
    def __init__(self, model_name, backend):
        self.model_name = model_name
        self.backend = backend
        self.number_of_classes = utils.get_number_of_classes()
        self.session = None
        self.input_names = None
        self.output_names = None
        self.input_height = 0
        self.input_width = 0
        self.input_tensor = None

        ## initializing model and create session
        self.initialize_model(self.model_name)


    def initialize_model(self, path):
        options = onnxruntime.SessionOptions()

        # (Optional) Enable configuration that raises an exception if the model can't be
        # run entirely on the QNN HTP backend.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
        self.session = onnxruntime.InferenceSession(path,
                                                    sess_options=options,
                                                    providers=["QNNExecutionProvider"] if self.backend == "npu" else ["CPUExecutionProvider"],
                                                    provider_options=[{"backend_path": "QnnHtp.dll" if self.backend == "npu" else "QnnCpu.dll"}])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    
    def prepare_input(self,image):
        img_height, img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # input_img = cv2.resize(input_img, (416, 416))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        self.input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32)
        #print(input_tensor.shape)
        # print(f"prepare_input time: {(time.perf_counter() - start)*1000:.2f} ms")
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def process_image(self, image):
        start = time.time()
        self.prepare_input(image)
        end = time.time()
        str1 = f"Preprocess Time: {(end-start)*1000:.3f} msecs"

        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_names[0]: self.input_tensor})
        end = time.time()
        str2 = f"Inference Time: {(end-start)*1000:.3f} msecs"
        
        start = time.time()
        softmax_outputs = self.softmax_2(outputs[0]).flatten()
        output = self.post_process_image(softmax_outputs)
        end = time.time()
        str3 = f"Post-Process Time: {(end-start)*1000:.3f} msecs"

        desc = "\r" + " ".join([str1, str2, str3]) + " " + str(output[0]) + " " + f"{output[1]:.5f}"
        print(desc, flush=True)
        return output

    def post_process_image(self, softmax_outputs):
        class_index = np.argmax(softmax_outputs)
        confidence_value = float(softmax_outputs[class_index])
        class_name = self.number_of_classes[class_index].split(',')[-1].strip()
        return class_name, confidence_value

    @staticmethod
    def softmax_2(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div
