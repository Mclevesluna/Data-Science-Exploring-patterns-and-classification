
# Data Science Project: Exploring patterns and classifications in 4 data sets

## Overview
This project aims to analyze the types of variables and data within four datasets, identifying relationships, patterns, and potential classification problems. To accomplish this, a series of steps to determine data/variable types, identify linearity and patterns, and explore classification problems were followed. For specific datasets, models such as decision trees, multiple regressions, and KNN predictors were also explored.

## Dataset
Data sets for this project are in folder titled "data" and were provided by Creative Computing Institute at University of the Art London (2023). 

## Preferred Language
Python

## Project Tasks

### 1. Data Quality Check
- **Objective**: Create a Data Pre-processing pipeline to ensure the dataset is ready for analysis.
- **Steps**:
  - Clean and preprocess the data.
  - Confirm the data quality and record the data shape.
  - Includes necessary comments and justifications throughout the process.

### 2. Data Relationship/Distribution Analysis
- **Objective**: Understand the distribution and relationships within the data.
- **Steps**:
  - Provide a Frequency table and plot to visualize Pickup counts by gender.
  - Provide Frequency tables and plots to visualize the distribution of Daily Average Minutes.
  - Analyze the relationship between:
    - Participant’s age and their Response time on singleton visual search.
    - Participant’s gender and their Response time on conjunction visual search.

### 3. Correlation Check
- **Objective**: Examine relationships between variables.
- **Steps**:
  - Produce a bivariate correlation table between Age, STAI, BRIEF_Total, DailyAvgMins, and VS_RT_correct_Single.

### 4. Linear Regression
- **Objective**: Assess if the minutes a person uses their mobile device per day predicts their visual search reaction time.
- **Steps**:
  - Perform a linear regression analysis.

### 5. Multiple Regression
- **Objective**: Evaluate the combined effect of multiple predictors on the outcome.
- **Steps**:
  - Add predictors (Age, Gender, Number of device pickups, etc.) to the regression model.
  - Determine if the variance accounted for in the outcome increases and if daily minutes of mbile usage remains a significant predictor.

### 6. Scenario 1 Analysis
- **Objective**: Test a hypothesis related to mobile usage and visual search performance.
- **Scenario**:
  - Participants were grouped by age and mobile usage.
  - They were asked to locate a target (red apple) among distractors (blue apples), and their reaction times were recorded.
- **Steps**:
  - Group participants and choose an appropriate Omnibus test statistic to test the hypothesis.
  - Justify the choice of test.
  - List assumptions and corresponding statistical tests.
  - Check and validate assumptions with visual charts.
  - Apply follow-on tests to identify specific effects.

### 7. Scenario 2 Analysis
- **Objective**: Test a hypothesis using a transformed dataset.
- **Scenario**:
  - Participants were asked to locate a target (red apple) among different distractors before and after a brain training exercise.
  - Their mobile usage was recorded and categorized.
- **Steps**:
  - Create groups and choose an appropriate Omnibus test statistic to test the hypothesis.
  - Justify the choice of test.
  - List assumptions and corresponding statistical tests.
  - Check and validate assumptions with visual charts.
  - Apply follow-on tests to identify specific effects.

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
from sklearn.svm import SVC![image](https://github.com/user-attachments/assets/9370ee22-3e8f-435e-8a80-b328b0e98e51)

3. **Run the Analysis**:
    Open the Jupyter notebook and run the cells to execute the analysis.
