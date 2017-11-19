# DM -EEG -Random Forest classification

# Calculating program run time
import time
start_time = time.clock()

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
 
# Setting the working directory
os.chdir("C:\\Users\\Aditya\\Desktop\\UTA Sems\\Sem_3\\Data Mining")

# Importing the dataset
dataset = pd.read_csv('FeatureMat_timeWin.csv', header = None)
X = dataset.iloc[:, 0:1344].values
y = dataset.iloc[:, 1344].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Creating and Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Implementing grid search algorithm
#from sklearn.model_selection import GridSearchCV
#parameters = [{'n_estimators': [5,10,20,50,100]}]
#grid_search = GridSearchCV(estimator = classifier, 
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 5,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

# Printing the program run time
print(time.clock() - start_time, "seconds")


