## Image Classification Inference

- This folder contains the script and required models and images to perform inference of Image Classification models using Python on Volterra
Dev Kit

### Pre-requisites:

- In order to run the application, we need to make sure the following pre-requisites are available
    - Python 3.11 (ARM64 version)
    - Onnxruntime-qnn
    - Opencv-python
    - Pillow
- Please make sure models are availble in [models](./assets/models/) folder, test images in [images](./assets/images/) and class names in [classes](./assets/imagenet_classes.txt)

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
python .\main.py
```
- This will run the image classification inference using **YOLO** model using **CPU** backend tested on **KEYBOARD** image. 
- If the application run successfully, we can able to see the output image with class label along with confidence value as shown below

- #### Options while running the application
    - #### image_path : input image for inference test
        - camera : for live video
        - <file_path> : image file path
        - ***Example Usage*** : 
            - python .\main.py --image_path camera
            - python .\main.py --image_path <file_path>
    - #### backend : backend selection for running inference script
        - cpu : float32 model running on cpu
        - npu : quantized model running on npu 
        - cpuqdq: quantized model running on cpu
        - ***Example Usage*** : 
            - python .\main.py --backend cpu
            - python .\main.py --backend npu
            - python .\main.py --backend cpuqdq
    - #### model : model selection for inference test
        - indeption : inception_v3 model
        - yolo : yolov8n-cls model
        - ***Example Usage*** : 
            - python .\main.py --model inception
            - python .\main.py --model yolo
