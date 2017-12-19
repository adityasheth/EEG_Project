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
os.chdir("E:\\Anger EEG\\Extra_data")

# Importing the dataset
#dataset = pd.read_csv('Complete_feature_vector_21_subjects.csv', header = None)
dataset = pd.read_csv('Feature_Vector_50_1.csv')
#X = dataset.iloc[:, 0:32].values # Feature vector
#y = dataset.iloc[:, 32].values # Target variable

X_Power = dataset[dataset.Feature == 1]
X_Power = X_Power.iloc[:,1:32].values
y_Power = dataset[dataset.Feature == 1]
y_Power = y_Power.iloc[:,32].values

#X_SNR = dataset[dataset.Feature == 2]
#X_SNR = X_SNR.iloc[:,0:32].values
#y_SNR = dataset[dataset.Feature == 2]
#y_SNR = y_SNR.iloc[:,32].values

#X_THD = dataset[dataset.Feature == 3]
#X_THD = X_THD.iloc[:,0:32].values
#y_THD = dataset[dataset.Feature == 3]
#y_THD = y_THD.iloc[:,32].values

#X_SINAD = dataset[dataset.Feature == 4]
#X_SINAD = X_SINAD.iloc[:,0:32].values
#y_SINAD = dataset[dataset.Feature == 4]
#y_SINAD = y_SINAD.iloc[:,32].values

#X = dataset.iloc[:,(3,4,5,6,9,11,14,16,21,22,24,26,27,29)].values # Important features as per backwards elimination in SAS
#dataset = pd.read_csv('Feature_Vector_100.csv', header = None)
#X = dataset.iloc[:, 0:30].values # Feature vector
#y = dataset.iloc[:, 30].values # Target variable

# Customized Standardizing
index = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315]
Scaled_values = pd.DataFrame(np.zeros(shape = (315,31)))
for k in range(0,21): # For 21 subjects
       start_1 = index[k]
       end_1 = index[k+1]
       indices_1 = list(range(start_1,end_1))
       indices_1 = np.array(indices_1)
       power_values = X_Power[start_1:end_1]
       power_values = pd.DataFrame(power_values) 
       Values = power_values.describe() # Extracting the important values from the boxplot
       Values = Values.iloc[:,:].values # Converting it to array/matrix form
       power_values = np.array(power_values)
       Scaled_values_1 = np.zeros(shape = (15,31))
       for j in range(0,31):
              IQR_25 = Values[4,j]
              IQR_75 = Values[6,j]
              IQR = IQR_75 - IQR_25
              V_U = IQR_75 + 1.5*IQR
              V_L = IQR_25 - 1.5*IQR
              Scaled_values_1[:,j] = (power_values[:,j] - V_L)/(V_U - V_L)
       Scaled_values[start_1:end_1] = Scaled_values_1

Scaled_values = np.array(Scaled_values)
#indexer = [0, 180, 360, 540, 720, 900, 1080, 1260]

indexer = [0, 45, 90, 135, 180, 225, 270, 315]
accuracy = np.zeros(shape=(7,1))

## Creating a loop for ensemble learning
#for k in range(0,4):
#       if k == 0
# Creating subject-wise k(7)-fold-cross-validation: 
for i in range(0,7):
       start = indexer[i]
       end = indexer[i+1]
       indices = list(range(start,end))
       indices = np.array(indices)
#       X_test = X[start:end]
#       X_train = np.delete(X, indices, axis=0)
#       y_test = y[start:end]
#       y_train = np.delete(y, indices, axis=0)
       X_test = Scaled_values[start:end]
       X_train = np.delete(Scaled_values, indices, axis=0)
       y_test = y_Power[start:end]
       y_train = np.delete(y_Power, indices, axis=0)
#       X_test = X_SNR[start:end]
#       X_train = np.delete(X_SNR, indices, axis=0)
#       y_test = y_SNR[start:end]
#       y_train = np.delete(y_SNR, indices, axis=0)
#       X_test = X_THD[start:end]
#       X_train = np.delete(X_THD, indices, axis=0)
#       y_test = y_THD[start:end]
#       y_train = np.delete(y_THD, indices, axis=0)
#       X_test = X_SINAD[start:end]
#       X_train = np.delete(X_SINAD, indices, axis=0)
#       y_test = y_SINAD[start:end]
#       y_train = np.delete(y_SINAD, indices, axis=0)
              
       # Feature Scaling
#       from sklearn.preprocessing import StandardScaler
#       sc = StandardScaler()
#       X_train = sc.fit_transform(X_train)
#       X_test = sc.transform(X_test)
       
       # Creating and Fitting classifier to the Training set
       from sklearn.ensemble import RandomForestClassifier
       classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
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

