from .utils import model_dir, image_dir, xywh2xyxy, nms, sigmoid, draw_detections
import onnxruntime
import cv2
import time
import numpy as np
import math
from colorist import BgColor


class HubYoloSeg:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32, backend='npu'):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.backend = backend

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)
    
    def segment_objects(self, image):
        start = time.perf_counter()
        input_tensor = self.prepare_input(image)
        end = time.perf_counter()
        print(f"Prepare input time: {(end - start)*1000:.5f} msecs")

        # Perform inference on the image
        start = time.perf_counter()
        outputs = self.inference(input_tensor)
        end = time.perf_counter()
        print(f"{BgColor.GREEN}Inference time: {(end - start)*1000:.5f} msecs{BgColor.OFF}")

        print([x.shape for x in outputs])
        print(outputs[3].shape)

        start = time.perf_counter()
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs)
        self.mask_maps = self.process_mask_output(mask_pred, outputs[-1])
        end = time.perf_counter()
        print(f"Post-process time: {(end - start)*1000:.5f} msecs")

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def process_box_output(self, box_output):
        # here, box_output is a list of len 5.
        # Size are, [(1, 8400, 4), (1, 8400), (1, 8400, 32), (1, 8400), (1, 32, 160, 160)]
        # box_output shape: (1, 116, 8400)

        # predictions = np.squeeze(box_output).T
        # predictions shape (8400, 116)
        # num_classes = box_output.shape[1] - self.num_masks - 4
        # 80

        boxes = np.squeeze(box_output[0])   # (8400, 4)
        scores = np.squeeze(box_output[1])  # 1, 8400
        masks_weights = np.squeeze(box_output[2])
        class_idx = np.squeeze(box_output[3])
        masks = np.squeeze(box_output[4])


        # Filter out object confidence scores below threshold
        # scores = np.max(scores) # (8400,)

        # predictions = predictions[scores > self.conf_threshold, :]
        ind = np.where(scores > self.conf_threshold)
        scores = scores[ind]
        

        if len(scores) == 0:
            return [], [], [], np.array([])

        # Get the class with the highest confidence
        class_ids = np.array(list(map(round, class_idx[ind]))).astype(np.int64)
        print(f"==>> class_ids.shape: {class_ids}")

        # Get bounding boxes for each object
        boxes = self.extract_boxes(boxes)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        mask_predictions = masks_weights[ind]

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]
    
    def extract_boxes(self, box_predictions):
        # (8400, 4)
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes
    
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

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

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)