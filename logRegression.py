#Logistic Regression


import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

import matplotlib

matplotlib.rcParams.update({'font.size': 15})

X_train=pd.read_csv('X_train.csv',index_col=False).reset_index()
Y_train=pd.read_csv('y_train.csv').iloc[:,-1]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(X_train)
X_test=pd.read_csv('X_test.csv', index_col=False).reset_index()
Y_test=pd.read_csv('Y_test.csv').iloc[:,-1]


aucList = []
print("\n\nLogistic Regression: ")
LR = LogisticRegression(C = 10, penalty='l2')
training_error = LR.fit(X_train, Y_train).score(X_train, Y_train)
predictions = LR.predict(X_test)
i = 0
for x in LR.coef_[0]:
	print("%s :  %f "%( (list(X_train)[i]) , x))
	i+=1

print("\ntraining error: " + str(training_error))
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))


probs = LR.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
aucList.append(auc)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='LR')
# show the plot
