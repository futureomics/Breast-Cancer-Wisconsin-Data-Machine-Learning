#Breast Cancer Wisconsin Data Machine Learning 


library(readr)

library(ggplot2)
library(plotly)
library(dplyr)
library(naniar)
library(tidyverse)
library(ggcorrplot) # finding the correlation with variables 
library(caTools)# splitting data into training set test set 
library(caret)


data_cancer <- read.csv("C:/Users/Lenovo/Downloads/data.csv")
head(data_cancer)
str(data_cancer)

#To visualize all the variable in the data frame
data_1 <- data_cancer %>%
  as.data.frame() %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value")
ggplot(data_1, aes(value)) +
  geom_density() +
  facet_wrap(~variable)

#M and B ## Lets convert this into numeric only

data_cancer$diagnosis <- factor(data_cancer$diagnosis, levels = c("M","B"), labels = c(0,1))

data_cancer$diagnosis <- as.character(data_cancer$diagnosis)

data_cancer$diagnosis <- as.numeric(data_cancer$diagnosis)

str(data_cancer)

data_cancer <- data_cancer %>% relocate(diagnosis,.after= fractal_dimension_worst)

#Visualising the correlation between datasets
r <- cor(data_cancer[,3:32])

round(r,2)

ggcorrplot(r)

#computing a matrix of correlation p-value
ggcorrplot(r, hc.order = TRUE, type = "lower",
           outline.col = "white",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"))


data_cancer <- data_cancer[,1:32]

#Visualising the missing values in the data using naniar
sum(is.na(data_cancer))

#Lets check whether every columns have no missing values
sapply(data_cancer,function(x)sum(is.na(x)))


#Spliting data into training set and test set
split = sample.split(data_cancer$diagnosis, SplitRatio = 0.75)

train_set = subset(data_cancer, split ==TRUE)

test_set = subset(data_cancer, split ==FALSE)

#Feature scaling on few columns :
train_set[, 2:5] = scale(train_set[ , 2:5])

test_set[, 2:5] = scale(test_set[ , 2:5])

#Feature scaling on few columns : colun 14 to colmn 15
train_set[, 14:15] = scale(train_set[ , 14:15])
test_set[, 14:15] = scale(test_set[ , 14:15])
#view(train_set)

#Feature scaling on few columns : colun 22 to colmn 25
train_set[, 22:25] = scale(train_set[ , 22:25])
test_set[, 22:25] = scale(test_set[ , 22:25])

#apply machine learning models
#Support Vector Machine Model
library(e1071)

regressor_svm <- svm(formula = diagnosis ~ ., 
                     data=train_set,
                     type = 'C-classification',
                     kernel = 'linear')

#Predicting the test set results
y_pred1 = predict(regressor_svm, newdata = test_set[-32])

#confusion matrix
cm = table(test_set [ , 32], y_pred1)
cm

cv <- trainControl(method="cv",
                   number = 5,
                   preProcOptions = list(thresh = 0.99), # threshold for pca preprocess
                   classProbs = TRUE,
                   summaryFunction = twoClassSummary)

#Apply KNN MODEL :
# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_predknn = knn(train = train_set[, 2:31],
                test = test_set[, 2:31],
                cl = train_set[, 32],
                k = 5,
                prob = TRUE)

# Making the Confusion Matrix
cmknn = table(test_set[, 32], y_predknn)
cmknn                   


#Apply NAIVE’S BAYES MODEL :
# install.packages('e1071')
library(e1071)
classifier_bayes = naiveBayes(x = train_set[,2:31],
                              y = train_set$diagnosis)

# Predicting the Test set results
y_pred_bayes = predict(classifier_bayes, newdata = test_set[,2:31])

# Making the Confusion Matrix
cm_bayes = table(test_set[, 32], y_pred_bayes)

cm_bayes


#Apply RANDOM FOREST MODEL :
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier_rf = randomForest(x = train_set[,2:31],
                             y = train_set$diagnosis,
                             ntree = 500)

# Predicting the Test set results
y_pred_rf = predict(classifier_rf, newdata = test_set[,2:31])

# Making the Confusion Matrix
cm_rf = table(test_set[, 32], y_pred_rf)
cm_rf

y_pred_rf  
