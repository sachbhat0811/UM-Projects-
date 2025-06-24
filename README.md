
# Unified Mentor Projects Repository

This repository contains the complete code, reports, workflows, and detailed documentation for all Unified Mentor projects undertaken. Each project demonstrates a unique application of machine learning or deep learning, covering both image and structured data tasks.

---

## Projects Included

### 1. **Animal Image Classification**

* **Objective:** Develop a Convolutional Neural Network (CNN) to classify animal species from input images.
* **Key Technologies:** TensorFlow, Keras, CNN, Transfer Learning (MobileNetV2)
* **Highlights:**

  * Achieved high validation accuracy using MobileNetV2
  * Experimented with alternative architectures including EfficientNet
  * Detailed evaluation timeline of model variations and attempted improvements
* **Files Provided:**

  * Model training notebooks
  * Model evaluation results
  * Project report

---

### 2. **Mobile Phone Price Prediction**

* **Objective:** Predict the price range of mobile phones based on their technical specifications.
* **Key Technologies:** Scikit-learn, XGBoost, Random Forest, Regression and Classification Models
* **Highlights:**

  * Performed extensive exploratory data analysis (EDA)
  * Applied feature engineering, outlier handling, and hyperparameter tuning
  * Achieved high accuracy using baseline and advanced models
* **Files Provided:**

  * Data preprocessing scripts
  * Model training code
  * Project report

---

### 3. **Vehicle Price Prediction**

* **Objective:** Predict the actual price of vehicles based on features like mileage, year, and specifications.
* **Key Technologies:** Scikit-learn, Regression Models, Random Forest, Feature Selection
* **Highlights:**

  * Complete data cleaning workflow, including outlier management and feature transformation
  * Feature selection using Recursive Feature Elimination (RFE)
  * Performed hyperparameter tuning for model optimization
* **Files Provided:**

  * Model training and testing notebooks
  * Dataset cleaning scripts
  * Final project report

---

### 4. **American Sign Language (ASL) Detection**

* **Objective:** Build a CNN to classify American Sign Language hand signs into English alphabets.
* **Key Technologies:** TensorFlow, Keras, CNN, GPU Acceleration via WSL2
* **Highlights:**

  * Implemented end-to-end CNN for ASL image classification
  * Configured TensorFlow GPU support via WSL2 (Ubuntu 22.04)
  * Detailed GPU setup guide and environment configuration
* **Files Provided:**

  * Training and evaluation notebooks
  * GPU integration workflow
  * Final project report

---

## Repository Structure

```text
├── Animal-Image-Classification/
│   ├── Notebooks
│   ├── Reports
|   ├── Datasets
│   
│
├── Mobile-Phone-Price-Prediction/
│   ├── Notebooks
│   ├── Reports
|   ├── Datasets
│   
│
├── Vehicle-Price-Prediction/
│   ├── Notebooks
│   ├── Reports
|   ├── Datasets
│   
│
├── ASL-Detection/
│   ├── Notebooks
│   ├── Reports
|   ├── Datasets
│
└── README.md
```

---

## How to Run the Code

Each project is fully self-contained within its folder.

* Refer to the individual project notebooks for detailed, step-by-step instructions.
* TensorFlow GPU support for the ASL project is configured via WSL2 and Ubuntu 22.04.

---

## Prerequisites

* Python 3.10
* TensorFlow (version depending on project)
* Scikit-learn
* XGBoost
* Jupyter Notebook
* WSL2 with Ubuntu 22.04 (for ASL Detection GPU support)

Refer to the individual project folders for specific environment setup and dependencies.

---

## Reports

Each project folder includes a detailed project report covering:

* Problem statement and goals
* Dataset details
* Data processing workflows
* Model selection and hyperparameter tuning
* Obstacles faced and solutions implemented
* Final model performance and evaluation metrics

---

## License

This repository is for educational and project submission purposes only. Contact the repository owner for permission if you wish to use the code or reports beyond personal learning.

---

Let me know if you would like me to prepare additional sections such as dataset sources, setup scripts, or contribution instructions.
