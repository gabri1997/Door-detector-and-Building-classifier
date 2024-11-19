# Door Detector and Building Classifier

## Overview
This project was developed as part of the **Computer Vision and Cognitive System** course during the **UNIMORE 2020/2021 academic year**. It explores building recognition and entrance detection through neural networks and image retrieval techniques. For detailed insights, please refer to the full project report.

---

## The Team 👥
- **Benedetta Fabrizi**  
  [LinkedIn](https://www.linkedin.com/in/benedetta-fabrizi-54b7971b0) | [GitHub](https://github.com/BerniRubble)  
- **Emanuele Bianchi**  
  [LinkedIn](https://www.linkedin.com/in/emanuele-bianchi240497/) | [GitHub](https://github.com/Manu2497)

---

## Project Goals 🔍

The aim of this work is twofold:
1. **Building Classification**: Classify building types using a neural network.  
2. **Door Detection**: Identify entrances in images using object detection.  

Additionally, the project incorporates **image retrieval** to compare alternative approaches to classification, applying image processing techniques and feature extraction.

---

## 1. Introduction
Neural networks have the potential to improve everyday life significantly and can assist individuals with specific needs. This project exemplifies such applications in computer vision.

---

## 2. Building Classification

### 2.1 Dataset for ResNet
Since no suitable dataset was available, we created a custom one by collecting 1,300 images from stock websites and organizing them into five classes:
- House  
- Flat complex  
- Church  
- Historical buildings/monuments  
- Shops  

To ensure balance, we curated the dataset so each class contained approximately the same number of images. 

### 2.2 Model Selection
We employed **ResNet** (Residual Network) for building classification. The final layer was modified to output a `[1, 5]` matrix, representing the five building classes. Fine-tuning was achieved by freezing the initial layers and retraining the last 19 layers.  

#### Experimentation
- Models tested: **ResNet18**, **ResNet34**, **ResNet50**, and **ResNet152**.  
- Optimization: Stochastic Gradient Descent (SGD) vs. Adam optimizer.  
- Best result: **ResNet34** with a learning rate of 0.005, optimized using SGD.  

---

## 3. Door Detection

The dataset from the building classification task was annotated for door positions using **LabelImg**. We employed **YOLOv2** (You Only Look Once) for door detection, fine-tuning it on approximately 1,000 images. YOLOv2 was chosen for its speed, though it is less accurate than region-proposal architectures.

---

## 4. Image Retrieval

### Approaches
Two methods for image retrieval were tested:
1. **ORB Features**: Using Oriented FAST and Rotated BRIEF descriptors.  
2. **Neural Network Features**: Extracting features from the ResNet's last layer before the fully connected layer.

### Dataset Processing
The images underwent preprocessing steps:
- Resizing to 400×400 pixels  
- Grayscale conversion  
- Histogram equalization  
- Bilateral filtering  

### Retrieval Methodology
Features were extracted, and similarity between query and dataset images was computed using **Mean Squared Error (MSE)**. Query expansion involved applying geometric transformations (rotation, perspective, WARP) to enhance feature matching.

---

## Results

### Key Findings:
1. **Building Classification**: ResNet34 achieved robust performance.  
2. **Door Detection**: YOLOv2 produced good results but showed room for improvement.  
3. **Image Retrieval**: Neural network features provided more stable results compared to ORB.  

#### Examples of Results:
- **Figure 6**: Good retrieval with ORB, ResNet, and YOLO; network-based retrieval failed.  
- **Figure 7**: Excellent door detection but incorrect retrieval and classification.  
- **Figure 8**: Accurate classification and retrieval; low door detection confidence.  
- **Figure 9**: Classification and network-based retrieval succeeded, but door detection and ORB retrieval failed.

---

## Example Outputs

### Example 1: Detection and Classification
![Three Doors](https://user-images.githubusercontent.com/58270634/190852798-8a9866e0-18ce-4ff7-955c-f71976e65831.jpg)

### Example 2: Image Retrieval and Detection
![Building and Door](https://user-images.githubusercontent.com/58270634/190852847-6dd7d641-636d-40cf-9bb7-b36dc598cc31.jpg)

---

## Future Improvements

1. Enhance door detection accuracy using YOLOv5 or YOLOv8.  
2. Increase dataset size and diversity for better generalization.  
3. Explore additional feature descriptors for image retrieval.  
4. Implement real-time deployment scenarios with optimized models.  
5. Extend the classification task to include additional building types.
