# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:53:05 2018

@author: Selam Mequanint
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ICU_df = pd.read_csv('ICU_df.csv')

print("ICU dead: " + str(ICU_df['hospital_expire_flag'].sum()))
print("ICU ALive: " + str(ICU_df.shape[0] - ICU_df['hospital_expire_flag'].sum() ))
y = ICU_df.hospital_expire_flag# define the target variable (dependent variable) as y

#drop the hospital expire flag from the data 
ICU_df.drop(['hospital_expire_flag', 'index', 'level_0', 'Unnamed: 0'], axis=1, inplace=True)
# create training and testing vars, 
X_train, X_test, Y_train, Y_test = train_test_split(ICU_df, y, test_size=0.2, random_state=43)
print("Xtrain:  "+ str(X_train.shape))
print("Ytrain:  "+ str(Y_train.shape))
print("Xtest:  "+ str(X_test.shape))
print("Ytest:  "+ str(Y_test.shape))

print("Xexpired: " + str(Y_train.sum()))
print("XAlive: " + str(Y_train.size - Y_train.sum() ))

print("Yexpired: " + str(Y_test.sum()))
print("YAlive: " + str(Y_test.size - Y_test.sum() ))

#SVM Starts here- with default first
clf = svm.SVC()
# Train classifier 
clf.fit(X_train, Y_train)
# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
## using linear
clf = svm.SVC(kernel='linear')
# Train classifier 
clf.fit(X_train, Y_train)
# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
#CV on linear kernel=this one took more than 40 min and I interapted it
clf=svm.SVC(kernel='linear')
scores = cross_val_score(clf, ICU_df, y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)
# Grid Search
# Parameter Grid
#5-fold cross validation, C=1 and gamma=1e-05 were selected as best parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, cv=5, verbose=10)
# Train the classifier
clf_grid.fit(X_train, Y_train)
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
###Non-linear kernel
clf = svm.SVC(kernel='rbf', C = 1.0, gamma=0.1)
# Train classifier 
clf.fit(X_train, Y_train)
# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

#Try Kernel=polynomial
clf = svm.SVC(kernel='poly')
# Train classifier 
clf.fit(X_train, Y_train)
# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

#Random forest (Precision=76%, recall=82%, f1-score=79%)
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rfc = RandomForestClassifier(n_estimators = 1000, random_state = 43)
# Train the model on training data
rfc.fit(X_train, Y_train)
# Use the forest's predict method on the test data
predictions = rfc.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))



