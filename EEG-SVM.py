# Kernel(RBF) SVM:

# Calculating process time
import time
start_time = time.clock()

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Setting the working directory
os.chdir("C:\\Users\\Aditya\\Desktop\\UTA Sems\\Sem_3\\Data Mining")

# Importing the dataset
dataset = pd.read_csv('FeatureMat_timeWin.csv', header = None)
X = dataset.iloc[:, 0:1344].values # Feature vector
y = dataset.iloc[:, 1344].values # Target variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 1, random_state = 0) # poly, sigmoid
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

## Implementing grid search algorithm
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [0.01,0.1,1,10,100], 'gamma': [0.1,0.2,0.5,1,5,10]}]
#grid_search = GridSearchCV(estimator = classifier, 
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 5,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

# Printing program runtime
print(time.clock() - start_time, 'seconds')
