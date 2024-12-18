import onnxruntime
import time
import cv2
import numpy as np
from colorist import BgColor


class SuperResolution:
    def __init__(self, path, backend="cpu", nchw=True, data_type='float'):
        self.backend = backend
        self.nchw = nchw
        self.dtype = np.uint8 if data_type == "uint8" else np.float32 

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        options = onnxruntime.SessionOptions()

        # (Optional) Enable configuration that raises an exception if the model can't be
        # run entirely on the QNN HTP backend.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = onnxruntime.InferenceSession(path,
                                                    sess_options=options,
                                                    providers=["QNNExecutionProvider"] if self.backend == "npu" else ["CPUExecutionProvider"],
                                                    provider_options=[{"backend_path": "QnnHtp.dll" if self.backend == "npu" else "Qnncpu.dll"}])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        start = time.perf_counter()
        input_tensor = self.prepare_input(image)
        print(f"prepare_input time: {(time.perf_counter() - start)*1000:.2f} ms")

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        if isinstance(outputs, list):
            outputs = outputs[0]

        start = time.perf_counter()
        outputs = self.post_process(outputs)
        print(f"Post process time: {(time.perf_counter() - start)*1000:.2f} ms")

        return outputs
    
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        # this is for single channels
        input_img = input_img / 255
        if self.nchw:
            input_img = input_img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_img, axis=0).astype(self.dtype)

        return input_tensor
    
    def post_process(self, image):
        if len(list(image.shape)) == 4:
            image = image[0]
        image = np.array(image).clip(0, 1) * 255
        if self.nchw:
            image = np.transpose(image, (1, 2, 0))
        image = np.array(image).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"{BgColor.GREEN}Inference time: {(time.perf_counter() - start)*1000:.2f} ms{BgColor.OFF}")
        return outputs
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2] if self.nchw else self.input_shape[1]
        self.input_width = self.input_shape[3] if self.nchw else self.input_shape[2]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]