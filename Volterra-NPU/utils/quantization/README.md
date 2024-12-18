<h3 align="center">QUANTIZATION</h3>

---

<p align="center"> Quantization of onnx to onnx qdq model
    <br> 
</p>

## 📝 Table of Contents

- [Python Packages](#problem_statement)
- [Installation](#installation)
- [Getting Started](#getting_started)
- [Prerequisites](#prerequisites)
- [Model Preprocessing](#model_preprocessing)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)

## Python Packages <a name = "python_packages"></a>

- onnxruntime==1.18.0
- numpy==1.25.2
- pillow

## Installation <a name = "installation"></a>

```
pip install onnxruntime
pip install numpy==1.25.2
pip install pillow
```

## 🏁 Getting Started <a name = "getting_started"></a>

* Put the onnx model in the root directory.
* Run the script.


## Prerequisites

```
Python3.11.x
```

## Model Preprocessing <a name = "preprocessing"></a>
Recommended but not Required.
```
python -m onnxruntime.quantization.preprocess --input xlsr.onnx --output xlsr.onnx
```

## Usage <a name = "usage"></a>

`
python main.py --model_name xlsr --from_tflite --remove_activations
`

This command with take the xlsr.onnx model and convert it into quantized version of the model.

## Acknowledgement <a name = "acknowledgement"></a>

Validation images are used to calculate the quantized value of the parameters during static quantization.

`
DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and CVPR 2018) and @ PIRM (ECCV 2018)
`
