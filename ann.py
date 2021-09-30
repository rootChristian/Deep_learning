#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:30:58 2021

@author: christiankemgang
"""

# Import libraries
import pandas as pd
import numpy as np

# Import data
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
Y = dataset.iloc[:, 13]

#Handling categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[
    ('Geography',OneHotEncoder(),[1]),
    ('Gender',OneHotEncoder(),[2])  ], remainder='passthrough')
X = ct.fit_transform(X)
#X = X[:, 1:]
X = np.delete(X, [0,3], 1)

# Divide the dataset between the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Import module keras
import keras
from keras.models import Sequential
from keras.layers import Dense
# Use to resolve overlearning problem
from keras.layers import Dropout
"""
# Inizializing
classifier = Sequential()

# Add input layer and hidden layer
classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", 
                     input_dim=11))

classifier.add(Dropout(rate=0.1))

# Add a second hidden layer
classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.1))

# Add input layer
classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))
# Compilation
classifier.compile(optimizer="adam", 
                   loss="binary_crossentropy", metrics=["accuracy"])

# Training the neural network
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
"""

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
"""
new_pred = classifier.predict(sc.transform(np.array([
    [0, 0, 1, 600, 40, 3, 60000, 2, 1, 1, 50000]
  ])))
new_pred = (new_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
"""


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
"""
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", 
                     input_dim=11))
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))
    classifier.compile(optimizer="rmsprop", 
                   loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=32, epochs=500)
precisions = cross_val_score(classifier, X=X_train, y=Y_train, cv=10, n_jobs=-1)
medium = precisions.mean()
standard_derivation = precisions.std()
"""

# K-fold cross validation
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", 
                     input_dim=11))
    classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer, 
                   loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size": [25, 32], "epochs": [100, 500], 
              "optimizer": ["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_


