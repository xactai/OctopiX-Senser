## Object Detection Inference

- This folder contains the script, and required models and images to perform inference of Image Segmentation models using Python on Volterra
Dev Kit 2023 (Snapdragon 8cx Gen 3)

### Pre-requisites:

- In order to run the application, we need to make sure the following pre-requisites are available
    - Python 3.11 (ARM64 version)
    - Onnxruntime-qnn
    - Opencv-python
    - Pillow
- Please make sure models are availble in [models](./assets/models/) folder and test images in [images](./assets/images/)

### Activating Virtualenv

- Before activating virtualenv please make sure virtualenv is created by using the command
```bash
virtualenv <virtualenv_name>
```
- Activate the virtualenv with the following command
```bash
<virtualenv_folder>\scripts\activate
```

### Running the Application
- Open command prompt and type the following
```bash
python .\image_segmentation_inference.py --model yolo --image_path chairs.jpg --backend cpu
```
- This will run the image segmentation inference using **yolov8n-seg** model using **CPU** backend tested on **CHAIRS** image. 
- If the application run successfully, we can able to see the output image with higher resolution.

- #### Options while running the application
    - #### image_path : input image for inference test
        - <file_path> : image file path
        - ***Example Usage*** : 
            - python .\image_segmentation_inference.py --image_path camera
            - python .\image_segmentation_inference.py --image_path <file_path>
            - python .\image_segmentation_inference.py --image_path rtsp
    - #### backend : backend selection for running inference script
        - cpu : float32 model running on cpu
        - npu : quantized model running on npu 
        - cpuqdq: quantized model running on cpu
        - ***Example Usage*** : 
            - python .\image_segmentation_inference.py --backend cpu
            - python .\image_segmentation_inference.py --backend npu
            - python .\image_segmentation_inference.py --backend cpuqdq
    - #### model : model selection for inference test
        - yolo : yolov8n-seg model
        - ffnet78 : ffnet-78s model
        - ffnet40 : ffnet-40s model
        - ***Example Usage*** : 
            - python .\image_segmentation_inference.py --model yolo
            - python .\image_segmentation_inference.py --model ffnet78
            - python .\image_segmentation_inference.py --model ffnet40
