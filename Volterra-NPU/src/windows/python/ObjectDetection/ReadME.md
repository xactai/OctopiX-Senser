## Object Detection Inference

- This folder contains the script and required models and images to perform inference of Object Detection models using Python on Volterra
Dev Kit

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
python .\image_detection_inference.py
```
- This will run the image classification inference using **YOLOv8** model using **CPU** backend tested on **CHAIRS** image. 
- If the application run successfully, we can able to see the output image with class label along with confidence value as shown below

- #### Options while running the application
    - #### image_path : input image for inference test
        - camera : for live video
        - <file_path> : image file path
        - <file_path> : video file path in the assets folder
        - ***Example Usage*** : 
            - python .\image_detection_inference.py --image_path camera
            - python .\image_detection_inference.py --image_path <file_path>
            - python .\image_detection_inference.py --image_path <video_path>
    - #### backend : backend selection for running inference script
        - cpu : float32 model running on cpu
        - npu : quantized model running on npu
        - cpuqdq: quantized model running on cpu
        - ***Example Usage*** : 
            - python .\image_detection_inference.py --backend cpu
            - python .\image_detection_inference.py --backend npu
            - python .\image_detection_inference.py --backend cpuqdq
    - #### model : model selection for inference test
        - yolov8 : yolov8n model
        - yolov11 : yolov11n model
        - ***Example Usage*** : 
            - python .\image_detection_inference.py --model yolov8
            - python .\image_detection_inference.py --model yolov11
            - python .\image_detection_inference.py --image_path sample.mp4 --model yolov8
            
            `It will save the processed file in the [output](./assets/output/) folder.`
