# Digit Classification using Scikit-Learn KNN

## Project Overview
This project implements a handwritten digit classifier using the **K-Nearest Neighbors (KNN)** algorithm from the **Scikit-learn** library. It demonstrates a complete machine learning pipeline, from data preprocessing to model evaluation and visualization.

## Workflow
- **Data Loading:** Load the Scikit-learn Digits dataset containing 8x8 images of handwritten digits.
- **Data Cleaning:** Handle missing values and remove duplicates while ensuring target alignment.
- **Feature Scaling:** Apply **StandardScaler** to normalize feature values, ensuring optimal performance for distance-based algorithms like KNN.
- **Label Encoding:** Process labels to ensure compatibility with the classifier.
- **Data Splitting:** Partition the dataset into **80% training** and **20% testing** sets.
- **Model Implementation:** Build and train the model using the optimized `KNeighborsClassifier` from Scikit-learn.
- **Model Evaluation:** Assess performance using:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-score)
  - **Confusion Matrix Visualization** (using Seaborn heatmaps for visual clarity)

## Objective
The goal of this project is to build an efficient and highly accurate classification model using industry-standard libraries, focusing on the practical application of KNN and data visualization techniques in a real-world dataset.

## Technologies Used
- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## How to Run
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn