# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:32:24 2018

@author: kmy07
"""
#Step 0: Import the neccessary Packages
import pandas as pd
import numpy as np
import time

#Step 1: Import the dataset
dataset = pd.read_csv(r'./dataset.csv')
print("Dataset Imported Successfully")

#Step 2: DataPreprocessing
columns = ['type','amount','oldbalanceOrg','isFraud']
datasets = pd.DataFrame(dataset,columns=columns)
print("Data PreProcessed")

#Step 3: Encoding for categorical Variables
datasets['type'] = pd.get_dummies(datasets['type'])


'''Check the Percentage of rows with negative class'''
li = datasets.loc[datasets['isFraud'] == 1]
print(len(li))

#Step 4: Split the data into test data and train data
from sklearn.model_selection import train_test_split

X = datasets.drop('isFraud',axis=1) 
Y = datasets['isFraud'] #target Colummn
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

#Step 5: Model Setup
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)

#Step 6: Prediction
y_pred = knn.predict(X_test)
np.mean(y_pred == Y_test)

#Step 7: Prediction for Custom Data
X_new_1 = np.array([[1,121212123,123]])
X_new_2 = np.array([[1,123,123]])

print("Validating first transaction...\n")
time.sleep(3)
result = knn.predict(X_new_1)
if result[0] == 1:
    print("The first transaction must be fraudulent transaction\n" )
else:
    print("The transaction is not a fraudulent transaction\n")
    
print("\nValidating second transaction...")
time.sleep(3)
result = knn.predict(X_new_2)
if result[0] == 1:
    print("The second transaction must be fraudulent transaction\n" )
else:
    print("The transaction is not a fraudulent transaction\n")