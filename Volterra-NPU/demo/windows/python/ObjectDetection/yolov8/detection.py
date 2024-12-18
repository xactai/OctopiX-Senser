import time
import cv2
import numpy as np
import onnxruntime

from yolov8.utils import xywh2xyxy, draw_detections, multiclass_nms


class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, backend="cpu"):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.backend = backend

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

        start = time.perf_counter()
        try:
            self.boxes, self.scores, self.class_ids, self.original_boxes = self.process_output(outputs)
        except:
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)
            self.original_boxes = None
        print(f"process_output time: {(time.perf_counter() - start)*1000:.2f} ms")

        return self.boxes, self.scores, self.class_ids, self.original_boxes


    def prepare_input(self, image):
        start = time.perf_counter()
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # input_img = cv2.resize(input_img, (416, 416))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32)
        # print(f"prepare_input time: {(time.perf_counter() - start)*1000:.2f} ms")

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes, original_boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], original_boxes[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # original boxes from yolo (460 x 460)
        original_boxes = np.copy(boxes)
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        original_boxes = xywh2xyxy(original_boxes)

        return boxes, original_boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]