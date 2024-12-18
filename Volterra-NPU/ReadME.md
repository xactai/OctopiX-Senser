<div align="center">
  <h1 style="font-size: 3rem; color: #2D89EF;">README FOR VOLTERRA</h1>
  <h2 style="color: #FF6F61;">VOLTERRA CPU-NPU-GPU</h2>

  <a href="#"><img src="https://img.shields.io/badge/status-active-success.svg" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</div>

<h3 align="center" style="color: #555;">AI Applications on Qualcomm Snapdragon 8cx Gen 3 CPU-NPU-GPU</h3>

---

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)

---

## 🧐 About <a name="about"></a>

This repository provides implementations in **Python**, **C++**, and **Java (Android)** for leveraging the **NPU, CPU, and GPU** of the **Microsoft Dev Kit 2023 (Project Volterra)**.  

### Applications Included:
- **Object Detection**  
  - [Python](src/windows/python/ObjectDetection/)  
  - [C++](src/windows/cpp/ObjectDetection/)  
- **Image Classification**  
  - [Python](src/windows/python/Classification/)  
  - [C++](src/windows/cpp/Classification/)  
  - [Android](src/android/ImageClassification/)  
- **Super Resolution**  
  - [Python](src/windows/python/SuperResolution/)  
  - [C++](src/windows/cpp/SuperResolution/)  
  - [Android](src/android/SuperResolution/)  
- **Semantic Segmentation**  
  - [Python](src/windows/python/SemanticSegmentation/)  
- **Speech Recognition**  
  - [Python](src/windows/python/Whisper/)  
  - [Android](src/android/SpeechRecognition/)  

---

## 🏁 Getting Started <a name="getting_started"></a>

Clone the repository. The source code is available in the [src](Volterra-NPU/src) directory for Windows and Android.  

The [utils](utils) folder contains:  
- Quantization scripts  
- TFLite to ONNX converter scripts  

Each tool has its own README for step-by-step guidance.  

### Prerequisites  

To use this repository, you'll need:  
- **Git**  
- **Visual Studio Code** (or any Python IDE)  
- **Visual Studio Community 2022**  
- **Android Studio** (Ladybug | 2024.2.1 or later)  
- **Python 3.11 Virtual Environment**  
- **Python 3.8 Virtual Environment** (for Windows Python Speech Recognition App)  
- [QNN SDK from Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)  

---

## 🎈 Usage <a name="usage"></a>

Each folder contains a detailed **README.md** with specific instructions.  

- **C++ Applications**: [binaries](/Volterra-NPU/binaries/cpp/)  