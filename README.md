EuroSAT Land Cover Classification using ResNet50

This project builds a deep learning model to classify land cover types from satellite images using the EuroSAT RGB dataset and transfer learning with **ResNet50**.

---

Dataset  
- Source: [EuroSAT (RGB)](https://www.kaggle.com/datasets/nilesh789/eurosat-rgb)  
- Classes: Industrial, Forest, Residential, River, Sea/Lake, Highway, etc.  
- Format: RGB satellite images (64x64 pixels)

---

Key Steps

### 1. Exploratory Data Analysis
- Class distribution (bar chart + pie chart)
- Sample images visualization
- Image dimensions check
- Duplicates detection (via image hashing)
- RGB Channel mean and standard deviation

### 2. Preprocessing
- Data split (Stratified) into training and testing sets (80/20)
- Image Augmentation using `ImageDataGenerator` (rotation, zoom, flips, etc.)

### 3. Model Building
- Transfer Learning using pretrained **ResNet50** model
- Added custom classification head with dense and dropout layers
- Training done in two phases:
  - Phase 1: Train head layers (frozen base)
  - Phase 2: Fine-tune full model

### 4. Evaluation
- Accuracy and F2 score
- Classification report & Confusion matrix
- Accuracy/Loss curves plotted

---

Libraries Used
- `TensorFlow / Keras`
- `OpenCV`
- `Pandas`, `Seaborn`, `Matplotlib`
- `scikit-learn`
- `tqdm`, `PIL`

---

 Output
- Trained model saved: `ResNet50_eurosat.h5`
- Compressed project files: `eurosat_project.zip`
- Class mapping saved as: `class_indices.npy`

---

Results  
Achieved high classification accuracy across 10+ land cover types using transfer learning.

---
