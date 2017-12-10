# EEG - XGBoost

# Calculating process time
import time
start_time = time.clock()

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting the working directory
os.chdir("E:\\Anger EEG\\EEG_project")


# Importing the dataset
dataset = pd.read_csv('Feature_Vector.csv', header = None)
X = dataset.iloc[:, 0:31].values # Feature vector
y = dataset.iloc[:, 31].values # Target variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
Accuracy = (cm[0,0] + cm[1,1] + cm[2,2])/len(y_pred)

# Applying k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
#accuracies.mean()
#accuracies.std()

# Prnting the program runtime
print(time.clock() - start_time, 'seconds')

