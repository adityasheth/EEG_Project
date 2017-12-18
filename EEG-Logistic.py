# EEG - Logistic Regression

# Calculating process time
import time
start_time = time.clock()

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting the working directory
os.chdir("E:\\Anger EEG\\Extra_data")

# Importing the dataset
dataset = pd.read_csv('Complete_feature_vector_21_subjects.csv', header = None)
X = dataset.iloc[:, 0:31].values # Feature vector
y = dataset.iloc[:, 31].values # Target variable

indexer = [0, 180, 360, 540, 720, 900, 1080, 1260]
accuracy = np.zeros(shape=(7,1))

# Creating subject-wise k(7)-fold-cross-validation: 
for i in range(0,7):
       start = indexer[i]
       end = indexer[i+1]
       indices = list(range(start,end))
       indices = np.array(indices)
       X_test = X[start:end]
       X_train = np.delete(X, indices, axis=0)
       y_test = y[start:end]
       y_train = np.delete(y, indices, axis=0)

       # Feature Scaling
       from sklearn.preprocessing import StandardScaler
       sc = StandardScaler()
       X_train = sc.fit_transform(X_train)
       X_test = sc.transform(X_test)
       
       # Fitting Logistic Regression to the Training set
       from sklearn.linear_model import LogisticRegression
       classifier = LogisticRegression(penalty = 'l1', random_state = 0)
       classifier.fit(X_train, y_train)
       
       # Predicting the Test set results
       y_pred = classifier.predict(X_test)
       
       # Making the Confusion Matrix
       from sklearn.metrics import confusion_matrix
       cm = confusion_matrix(y_test, y_pred)
       accuracy[i] = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_pred)

# Printing the accuracy, standard deviation and program run time:
print("\nMean accuracy =", accuracy.mean()*100,"%")
print("\nStandard deviation =",accuracy.std()*100,"%")
print("\nProcessing time =", time.clock() - start_time, "seconds")