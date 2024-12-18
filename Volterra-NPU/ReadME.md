<div align="center">
  <h1>
    README FOR VOLTERRA 
  </h1>

  <h2>VOLTERRA CPU-NPU-GPU</h2>

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

</div>


<h3 align="center"> AI applications on Qualcomm Snapdragon 8cx Gen 3 CPU-NPU-GPU</h3>

---

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)


## 🧐 About <a name = "about"></a>

This repository is to provide the code in Python, C++ and Java(Android), for windows and android, to get the leverage of NPU, CPU and GPU of Microsoft Dev Kit 2023 (Project Volterra).

The application included in this repository are:
- Object Detection - [Python](src/windows/python/ObjectDetection/) [C++](src/windows/cpp/ObjectDetection/)
- Image Classification - [Python](src/windows/python/Classification/) [C++](src/windows/cpp/Classification/) [Android](src/android/ImageClassification/)
- Super Resolution - [Python](src/windows/python/SuperResolution/) [C++](src/windows/cpp/SuperResolution/) [Android](src/android/SuperResolution/)
- Semantic Segmentation - [Python](src/windows/python/SemanticSegmentation/) 
- Speech Recognition - [Python](src/windows/python/Whisper/) [Android](src/android/SpeechRecognition/)

## 🏁 Getting Started <a name = "getting_started"></a>

Clone the repository. The source code is available in [src](Volterra-NPU/src) directory for windows and android.

The [utils](utils) folder is containing the quantization scripts, as well as the tflite to onnx converter scripts with the README.

### Prerequisites

* Git
* Visual Studio Code (or any other Python IDE)
* Visual Studio Community 2022
* Android Studio >= Ladybug | 2024.2.1
* Python3.11 Virtual Environment
* Python3.8 Virtual Environment (For Windows Python Speech Recognition App)
* [QNN SDK from Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)


## 🎈 Usage <a name="usage"></a>

Respective folders has its own **README.md** file with the instructions.

- Python applications are in [src/windows/python](src/windows/python/) directory.

- C++ applications are in [src/windows/cpp](src/windows/cpp) directory.

- Android applications are in [src/android](src/android) directory.

