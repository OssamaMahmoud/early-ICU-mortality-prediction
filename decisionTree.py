#decisionTree


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


X_train=pd.read_csv('X_train.csv',index_col=False)
Y_train=pd.read_csv('y_train.csv').iloc[:,-1]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(X_train)
X_test=pd.read_csv('X_test.csv', index_col=False)
Y_test=pd.read_csv('Y_test.csv').iloc[:,-1]

print("\n\nDecision Tree: ")
DT = DecisionTreeClassifier(max_depth=6)
training_error = DT.fit(X_train, Y_train).score(X_train, Y_train)
predictions = DT.predict(X_test)
print("\ntraining error: " + str(training_error))
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))

dot_data = StringIO()
export_graphviz(DT, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=list(X_train),
                label='none', impurity=False, proportion=False )

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("iris.png")


probs = DT.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
aucList.append(auc)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='DT')
# show the plot
