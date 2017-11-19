# ANN - EEG

# Using Artificial Neural network to predict the category of EEG data
# Number of categories = 4
# Using 2 hidden layers. Increased number of hidden layers tend to reduce accuracy and also increase computing time
# Number of neurons = 672 (total number of features/2)(After optimizing reduced it to 50)
# Activation function = Hinge(relu)
# Output layer activation function is "softmax", since as this is a multi-class classification problem

# Calculating process time
import time
start_time = time.clock() # Starting the clock

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
 
# Setting the working directory
os.chdir("C:\\Users\\Aditya\\Desktop\\UTA Sems\\Sem_3\\Data Mining")

# Importing the dataset
dataset = pd.read_csv('FeatureMat_timeWin.csv', header = None)
X = dataset.iloc[:, 0:1344].values # Feature vector
y = dataset.iloc[:, 1344].values # Target variable

# Encoding categorical data
import keras.utils
y = keras.utils.to_categorical(y)
y = y[:,1:] # Coverted the 4 categories into binary system

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD

# def baseline_model():
# Initializing the ANN
model = Sequential()
      
# Adding the input layer and the first hidden layer
model.add(Dense(activation = 'relu', units = 50, input_dim = 1344, kernel_initializer = 'uniform'))
       
# Adding the second hidden layer
model.add(Dense(activation = 'relu', units = 50, kernel_initializer = 'uniform'))

# Adding a dropout
model.add(Dropout(0.5))

# Adding the output layer
model.add(Dense(activation = 'softmax', units = 4, kernel_initializer = 'uniform'))
       
# Compiling the ANN
# sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
# Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999) # Parameters used in paper are same as default parameters
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#return model

# Fitting the ANN to the training set
# estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
model.fit(X_train, y_train, epochs=5, batch_size=20)
# Part 3 -Making predictions and evaluating the model

# Predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)

# Decoding the target variable
yhat = np.zeros(shape=(len(y_pred),1))
i = 0
for i in range(0,len(y_pred)):
       if y_pred[i,0] == 1:
              yhat[i] = 1
       elif y_pred[i,1] == 1:
              yhat[i] = 2
       elif y_pred[i,2] == 1:
              yhat[i] = 3
       else:
              yhat[i] = 4
i = i+1

y_test_new = np.zeros(shape=(len(y_test),1))
y_test = y_test.astype(int)
j = 0       
for j in range(0,len(y_test)):
       if y_test[j,0] == 1:
              y_test_new[j] = 1
       elif y_test[j,1] == 1:
              y_test_new[j] = 2
       elif y_test[j,2] == 1:
              y_test_new[j] = 3
       else:
              y_test_new[j] = 4
j = j+1

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_new, yhat)

def Accuracies(cm,y_test_new): # Defining accuracies function
       acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/len(y_test_new)
       return acc

acc = Accuracies(cm,y_test_new) # Gives the accuracy

# Applying k-fold cross validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring = 'accuracy')
#accuracies.mean()
#accuracies.std()
#cross_val_score()

# Implementing grid search algorithm
#from sklearn.model_selection import GridSearchCV
#parameters = [{'epochs': [1,5,10,20,50], 'activation': ['tanh'], 'units': [10,100,200,500]}]
#grid_search = GridSearchCV(estimator = model, 
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 5,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

# Printing the program runtime
print (time.clock() - start_time, "seconds")




