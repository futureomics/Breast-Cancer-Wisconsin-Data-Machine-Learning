# Breast-Cancer-Wisconsin-Data-Machine-Learning
Breast Cancer Wisconsin Data Machine Learning

**Overview**

The Breast Cancer Wisconsin dataset is a commonly used dataset for machine learning classification tasks.
This project demonstrates how to apply machine learning techniques in R to classify breast cancer diagnoses using the dataset.

**Dataset Information**

**Source: UCI Machine Learning Repository**
**Features:** 30 numerical features (e.g., radius, texture, smoothness, compactness, symmetry, etc.)

**Target Variable:** Diagnosis (Malignant = M, Benign = B)

**Sample Size**: 569 instances

**Requirements**

To run this project, install the following R packages:
install.packages(c("tidyverse", "caret", "randomForest", "e1071"))

**Steps to Implement Machine Learning in R**

**1. Load Necessary Libraries**

library(tidyverse)
library(caret)
library(randomForest)
library(e1071)

