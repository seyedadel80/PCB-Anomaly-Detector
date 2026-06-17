# ⚙️ Autonomous PCB Quality Inspection System

### Industrial-Grade Anomaly Detection & Defect Segmentation using Deep Learning

This repository features a complete, end-to-end Computer Vision pipeline designed to automate quality control on Printed Circuit Boards (PCBs). By leveraging a Deep Semantic Segmentation framework, the system detects, isolates, and highlights manufacturing defects (such as short circuits, missing traces, or solder bridges) through an interactive web-based interface.

---

## 🖥️ User Interface Preview (Gradio Web App)

The production-ready application is powered by **Gradio**, providing an intuitive interface for quality control inspectors. It allows users to upload high-resolution PCB images, adjust defect confidence thresholds in real-time, and instantly view localized anomalies.

### 1. Normal PCB Inspection (False-Alarm Prevention)

When a flawless PCB is uploaded, the system processes the image through the network and ensures that no pixel probability exceeds the set threshold. It outputs a clean status with a low global anomaly score, minimizing false positives.

<p align="center">
  <img src="ui_normal.png" width="85%" alt="Normal PCB Detection State" />
</p>

### 2. Anomaly Detection & Visual Fault Isolation

When a defective board is inspected, the model generates a dense pixel-wise probability map. The post-processing pipeline detects active contours and draws automated bounding boxes directly around the faulty regions, providing immediate visual feedback.

<p align="center">
  <img src="ui_anomaly.png" width="85%" alt="Anomaly Detected State with Bounding Boxes" />
</p>

---

## ✨ Core Features

* **Deep Learning Segmentation:** Leverages a **U-Net** architecture with a **ResNet34** encoder (via `segmentation_models_pytorch`) for high-precision defect localization.
* **Interactive UI:** Built with **Gradio**, allowing inspectors to upload PCB images, adjust defect confidence thresholds in real-time, and view binary masks.
* **Visual Fault Isolation:** Highlights detected anomalies automatically with clean, red bounding boxes directly on the original board image.
* **Recent History Tracking:** Saves and displays the last 5 inspected images within the session, enabling quick click-to-reload historical comparisons.

---

## 🧠 Methodology & Model Architecture

The core of this system is built upon **Semantic Segmentation**, treating defect detection not just as a global classification problem, but as a pixel-level identification task.

* **Architecture:** **U-Net** (Encoder-Decoder framework optimized for capturing both high-level semantic context and precise low-level spatial localization).
* **Backbone / Encoder:** **ResNet34** (Pre-trained weights are leveraged to extract rich feature hierarchies from the PCB surfaces efficiently).
* **Activation:** Pixel-level probability mapping is achieved via a **Sigmoid** layer outputting continuous values between `0.0` (Normal) and `1.0` (Defect).
* **Post-Processing Pipeline:** Binary masks generated from the threshold slider are converted into spatial coordinates using OpenCV's topological contour algorithm (`cv2.findContours` with `RETR_EXTERNAL`), isolating each defect into structural bounding boxes.

---

## 📊 Dataset, Augmentation & Model Training

The training workflow was implemented inside Google Colab using an **NVIDIA T4 GPU**. The data preparation pipeline emphasizes robustness against varying factory lighting conditions and camera angles.

### 1. Dataset Sample & Ground Truth Target Mask

To train the segmentation network, each raw PCB image is mapped against a binary ground-truth mask. The image below (`dataset_sample.png`) displays a sample pair from the dataset where defects are precisely labeled in white against a black background, establishing the absolute ground truth for supervision.

<p align="center">
  <img src="dataset_sample.png" width="75%" alt="Dataset Ground Truth Mask Pair" />
</p>

### 2. Augmentation Pipeline & Preprocessing (`Albumentations`)

Due to the critical nature of industrial anomalies, the model utilizes the `Albumentations` library to apply real-time pixel and spatial augmentations. As shown in (`model_result.png`), input samples undergo geometric resizing, normalization, and advanced augmentations to ensure the model remains invariant to rotation, translation, and minor camera noise.

<p align="center">
  <img src="model_result.png" width="75%" alt="Data Preprocessing and Augmentation Pipeline" />
</p>

### 3. Training Convergence & Optimization Metrics

The network was trained using a combined loss function tailored for unbalanced segmentation data (where defects occupy a tiny fraction of the board). As analyzed in the metric curves below (`train_loss.png`), the training and validation loss converged stably over the epochs, ensuring high dice-coefficients and strict generalization without overfitting.

<p align="center">
  <img src="train_loss.png" width="80%" alt="Training and Validation Loss Optimization Curves" />
</p>

---

## 📂 Repository Structure

* `main.py`: The production deployment script hosting the inference engine, preprocessing blocks, and the Gradio UI layout.
* `PCBAnomalyDetector_Model.ipynb`: The exhaustive Google Colab notebook documenting dataset preprocessing, model training, and weight serialization.
* `requirements.txt`: List of precise Python package dependencies required to mirror the runtime environment.

---

## 🛠️ Installation & Local Setup

Deploy and run the interactive PCB quality inspection workspace locally by following these steps:

### 1. Clone the Workspace

First, clone this repository to your local drive and navigate into the project root directory:

```bash
git clone https://github.com/YOUR_USERNAME/PCB-Anomaly-Detection.git
cd PCB-Anomaly-Detection
```

### 2. Install Environment Dependencies

Ensure you have Python 3.8 or higher ready. Install all essential Deep Learning and Vision packages via `pip`:

```bash
pip install -r requirements.txt
```

> *GPU Note: To accelerate inference using an NVIDIA graphics card, verify your PyTorch setup aligns with your local CUDA driver version.*

### 3. Download & Position Model Weights (Critical Step)

Deep learning model weight files (`.pth`) are too heavy to be tracked directly inside GitHub repositories.

1. Run and execute the training blocks in the provided `PCBAnomalyDetector_Model.ipynb` notebook on Google Colab.
2. At the final cell, download the trained weights file named `best_pcb_model.pth`.
3. Move/Paste the downloaded `best_pcb_model.pth` file directly into the **root folder** of this project (in the same directory alongside `main.py`).

### 4. Launch the Web Application

Execute the production script to spin up the local server hosting the user interface:

```bash
python main.py
```

Upon execution, a local URL link (typically `http://127.0.0.1:7860`) will appear in your terminal. Click or open this link in any browser to launch your smart inspection workflow!
````
