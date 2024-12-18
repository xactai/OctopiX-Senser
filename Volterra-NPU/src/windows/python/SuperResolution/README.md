## Object Detection Inference

- This folder contains the script, and required models and images to perform inference of Super Resolution models using Python on Volterra
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
python .\main.py --model real_esrgan_x4plus --image_path Doll.jpg --backend cpu
```
- This will run the super resolution inference using **RealESRGAN** model using **CPU** backend tested on **Doll** image. 
- If the application run successfully, we can able to see the output image with higher resolution in the [outputs](.\outputs) folder.

- #### Options while running the application
    - #### image_path : input image for inference test
        - <file_path> : image file path
        - ***Example Usage*** : 
            - python .\main.py 
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
        - real_esrgan_x4plus : real_esrgan_x4plus model
        - xlsr : xlsr model
        - ***Example Usage*** : 
            - python .\main.py --model real_esrgan_x4plus
            - python .\main.py --model xlsr
