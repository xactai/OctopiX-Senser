<h3 align="center">TFLITE to ONNX Conversion</h3>

---

<p align="center"> Conversion of TFLITE model into ONNX model.
    <br> 
</p>

## 📝 Table of Contents

- [Python Packages](#problem_statement)
- [Installation](#installation)
- [Getting Started](#getting_started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)

## Python Packages <a name = "python_packages"></a>

- tf2onnx
- onnxruntime==1.18.0
- numpy==1.25.2

## Installation <a name = "installation"></a>

```
pip install tf2onnx
pip install onnxruntime
pip install numpy==1.25.2
```

## 🏁 Getting Started <a name = "getting_started"></a>

* Convert the TFLITE model into ONNX model.
* Quantize the ONNX into ONNX QDQ model.
* Use that model.


### Prerequisites

```
Python3.11.x
```

## Usage <a name = "usage"></a>
`
python -m tf2onnx.convert --tflite path_to_tflite_model --output path_to_output_onnx_model --opset 13
`
**opset 13 <= version <= 16**

