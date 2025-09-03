# A complete project for product category prediction based on titles 

This project implements a **machine learning pipeline** for automatic classification of product titles into predefined categories (e.g., CPUs, Mobile Phones, Fridges, Dishwashers, etc.).  
The best-performing model was **Support Vector Machine (SVM)**, which achieved high precision, recall, and F1-score across all categories.

# Project Structure

├── data/ 

│ └── products.csv #dataset

├── notebooks/

│ └── product_category_classification.ipynb # EDA and preprocessing

├── train_model.py # Script for training and saving the model

├── predict_category.py # Script for testing saved model with interactive prediction

└── README.md # Project documentation

## What We Did in This Module

Throughout this module, we covered all major steps of real-world ML project:

### 1. Project Setup

- Created a new GitHub repository
- Defined project folder structure
- Uploaded raw dataset

### 2. Data Exploration

- Loaded and analyzed a large dataset with products
- Used matplotlib and seaborn for visualizations
- Investigated the relationship between the Product title and Category Label columns

### 3. Data Cleaning & Preprocessing

- Removed missing values
- Standardized Category Labels column

### 4. Feature Engineering

- Selected meaningful input feature: Product Title, title_length and word_count
- Removed irrelevant columns

### 5. Model Training & Evaluation

- Compared multiple ML models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM)
- Used ColumnTransformer and Pipeline for unified preprocessing
- Evaluated using precision, recall, F1-score and confusion matrix

### 6. Model Training

- Trained final model on full dataset
- Saved the pipeline using **joblib** to **category_model.pkl**

### 7. Inference & Usage

- Loaded saved model
- Built an interactive interface for predicting category with new product title
- Enabled real time testing via console input

## How to Use

### Train the Model

---bash
python train_model.py
---

This will create a file called category_model.pkl in the model directory.

### Run Inference
Use the interactive script (predict_category.py) to classify new product title using the trained model.

## Author
This repository was developed as part of an educational program on practical machine learning using Python.
All steps were carefully documented and modularized to help users understand and reproduce the entire workflow.

## License
This project is open-source and freely available for educational use.
