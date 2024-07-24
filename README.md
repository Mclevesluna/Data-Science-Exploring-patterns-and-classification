
# Data Science Project: Exploring patterns and classifications in 4 data sets

## Overview
This project aims to analyze the types of variables and data within four datasets, identifying relationships, patterns, and potential classification problems. To accomplish this, a series of steps to determine data/variable types, identify linearity and patterns, and explore classification problems were followed. For specific datasets, models such as decision trees, multiple regressions, and KNN predictors were also explored.

## Dataset
Data sets for this project are in folder titled "data" and were provided by Creative Computing Institute at University of the Art London (2023). 

## Preferred Language
Python

## Project Tasks

### 1. Dataset Analysis
- **Objective**: Identify the type of each dataset and determine possible tasks.
- **Steps**:
  - **Dataset Type Identification**:
    - Determine if the dataset is Linear/NonLinear and Single/Multilabel.
    - Identify whether the tasks possible are Classification or Regression.
  - **Justification**:
    - **Variable Type Inspection**: Examine the dataset and report on the types of variables based on domain knowledge.
    - **Exploratory Analysis**: Perform at least one exploratory analysis technique (e.g., descriptive statistics, visualizations).
    - **Inferential Analysis**: Conduct one inferential analysis technique (e.g., hypothesis testing, confidence intervals).
    - **Predictive Analysis**: Apply one predictive analysis technique (e.g., regression model, classification algorithm).

### 2. Application of Loss Functions
- **Objective**: Apply various loss functions to appropriate datasets and compare their effectiveness.
- **Steps**:
  - **Loss Function Application**:
    - Apply the following loss functions as applicable:
      - L1 loss
      - L2 loss
      - Log loss
      - Categorical cross-entropy loss
      - Hinge loss
  - **Comparison**:
    - Create visual plots to compare the performance of different loss functions where applicable.

### 3. Metric Assessment
- **Objective**: Evaluate model performance using appropriate metrics.
- **Steps**:
  - **Metrics Evaluation**:
    - For Classification: Use metrics such as accuracy, precision, recall, F1 score, confusion matrix.
    - For Regression: Use metrics such as R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE).
  - **Documentation**:
    - Record and interpret the metrics for each dataset.

### 4. Non-Linear Dataset Transformation
- **Objective**: Transform a non-linear dataset to linear space and assess model performance.
- **Steps**:
  - **Kernel Transformation**:
    - Choose a non-linear dataset and apply a kernel transformation to linear space (e.g., Polynomial Kernel, Radial Basis Function).
  - **Model Fitting**:
    - Fit a model to the transformed data and evaluate its accuracy.

### 5. Regression Overfitting Analysis
- **Objective**: Create and analyze overfitting scenarios in regression.
- **Scenario**:
  - **Overfitting Creation**:
    - Adjust feature subsets or training dataset size to create overfitting conditions.
  - **Evidence**:
    - Use metrics and plots to demonstrate overfitting.
  - **Regularization**:
    - Apply two regularization methods (e.g., L1 regularization, L2 regularization) and evaluate model performance before and after regularization.

### 6. Classification Overfitting Analysis
- **Objective**: Create and analyze overfitting scenarios in classification.
- **Scenario**:
  - **Overfitting Creation**:
    - Adjust feature subsets or training dataset size to create overfitting conditions.
  - **Evidence**:
    - Use metrics and plots to demonstrate overfitting.
  - **Regularization**:
    - Apply two regularization methods (e.g., Dropout, L2 regularization) and evaluate model performance before and after regularization.

### 7. Decision Tree Analysis
- **Objective**: Apply Decision Tree models to specific datasets.
- **Steps**:
  - **Decision Tree Application**:
    - Apply a Decision Tree model to the following datasets:
      - MASTER_PhonesmartdataAll_CCI_AdvStats.csv
      - wine dataset
  - **Pruning**:
    - Apply pruning techniques to the Decision Tree models and record observations.

## Instructions for Running the Analysis

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Mclevesluna/Mobile-usage-and-search-performance.git
    cd your-repository-name
    ```

2. **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```
    Note: A GPU is not required for this project. It was developed on MacOS, ensuring compatibility with this operating system. However, it should work on other operating systems as well, though minor adjustments might be necessary.

Libraries used: 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

3. **Run the Analysis**:
    Open the Jupyter notebook and run the cells to execute the analysis.
