#logitic regression and random forest feature importance.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import neighbors
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn import svm
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

X_train=pd.read_csv('X_train.csv',index_col=False)
Y_train=pd.read_csv('y_train.csv').iloc[:,-1]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(X_train)
X_test=pd.read_csv('X_test.csv', index_col=False)
Y_test=pd.read_csv('Y_test.csv').iloc[:,-1]




print("\n\nLogistic Regression: ")
LR = LogisticRegression(C = 10, penalty='l2')
training_error = LR.fit(X_train, Y_train).score(X_train, Y_train)
predictions = LR.predict(X_test)




# Get numerical feature importances
importances = list(LR.coef_[0])
# List of tuples with variable and importance
feature_importances = [(feature, importance) for feature, importance in zip(list(X_train), importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: abs(x[1]), reverse = True)
# Print out the feature and importances
[print('{:20} : {}'.format(*pair)) for pair in feature_importances];


indices = np.argsort(importances)[::-1]


print('\nTop 10:')
[print('{:20} : {}'.format(*pair)) for pair in feature_importances[:10]];

featureName, featureImp = zip(*feature_importances[:10])
plt.subplot(1,2,1)
plt.bar(range(10),  list(map(abs,list(featureImp))), width=1.0,  align="center" , edgecolor='black')
plt.xticks(range(10), featureName, rotation = 30, ha="right")
plt.xlim([-1, 10])


print("\n\nRandom  Forest: ")
rf = RandomForestClassifier(n_estimators = 100, max_depth=6, n_jobs=4, oob_score=True)
training_error = rf.fit(X_train, Y_train).score(X_train, Y_train)

importances = list(rf.feature_importances_)
feature_importances = [(feature, importance) for feature, importance in zip(list(X_train), importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: abs(x[1]), reverse = True)
# Print out the feature and importances
[print('{:20} : {}'.format(*pair)) for pair in feature_importances];


indices = np.argsort(importances)[::-1]


print('\nTop 10:')
[print('{:20} : {}'.format(*pair)) for pair in feature_importances[:10]];

featureName, featureImp = zip(*feature_importances[:10])
plt.subplot(1,2,2)
plt.bar(range(10),  list(map(abs,(featureImp))), width=1.0,  color='red', align="center" , edgecolor='black')
plt.xticks(range(10), featureName, rotation = 30, ha="right")

plt.show()
