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

**2. Load Dataset**

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

dataset <- read.csv(url, header = FALSE)

**3. Data Preprocessing**

colnames(dataset) <- c("ID", "Diagnosis", paste("Feature", 1:30, sep="_"))

dataset$Diagnosis <- factor(dataset$Diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

dataset <- dataset[, -1]  # Remove ID column

**4. Train-Test Split**

set.seed(123)

trainIndex <- createDataPartition(dataset$Diagnosis, p = 0.8, list = FALSE)

trainData <- dataset[trainIndex, ]

testData <- dataset[-trainIndex, ]

**5. Model Training (Random Forest)**

rf_model <- randomForest(Diagnosis ~ ., data = trainData, ntree = 100)

**6. Model Evaluation**

predictions <- predict(rf_model, testData)

confusionMatrix(predictions, testData$Diagnosis)

**Results & Conclusion**

The model is evaluated using a confusion matrix to measure accuracy, sensitivity, and specificity.

Further improvements can be achieved by tuning hyperparameters, testing other algorithms (SVM, logistic regression, etc.), and feature selection.

**Acknowledgments**

UCI Machine Learning Repository

The R community for various packages and documentation

**License**

This project is open-source and available for use and modification under the MIT License.
