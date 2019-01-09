#full analysis
# using 6 machine learning methods
# calculates the accuracy on the test set, the fscore and the confusion matrix
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



def printResults(Y_test, predictions):
	print("\nTraining error: " + str(training_error))
	print("Accuracy: "+str(accuracy_score(Y_test, predictions)))
	print(confusion_matrix(Y_test, predictions))
	print(classification_report(Y_test, predictions))
	print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))


X_train=pd.read_csv('X_train.csv',index_col=False)
Y_train=pd.read_csv('y_train.csv').iloc[:,-1]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# 	print(X_train)
X_test=pd.read_csv('X_test.csv', index_col=False)
Y_test=pd.read_csv('Y_test.csv').iloc[:,-1]


matplotlib.rcParams.update({'font.size': 15})




aucList = []
print("\n\nLogistic Regression: ")
LR = LogisticRegression(C = 10, penalty='l2')
training_error = LR.fit(X_train, Y_train).score(X_train, Y_train)
predictions = LR.predict(X_test)
printResults(Y_test, predictions)


print("\n\nDecision Tree: ")
DT = DecisionTreeClassifier(max_depth=6)
training_error = DT.fit(X_train, Y_train).score(X_train, Y_train)
predictions = DT.predict(X_test)
printResults(Y_test, predictions)



print("\n\nRandom  Forest: ")
rf = RandomForestClassifier(n_estimators = 100, max_depth=6, n_jobs=4, oob_score=True)
training_error = rf.fit(X_train, Y_train).score(X_train, Y_train)
threshold = 0.4

predictions = rf.predict(X_test)
#predictions = (predicted_proba [:] >= threshold).astype('int')
printResults(Y_test, predictions)




print("\n\nNeural net: ")
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(9, 7))
training_error = mlp.fit(X_train,Y_train).score(X_train, Y_train)
predictions = mlp.predict(X_test)
printResults(Y_test, predictions)


print("SVM")
clf=svm.SVC(kernel='linear', probability=True, C=1, gamma=1e-05)
training_error = clf.fit(X_train,Y_train).score(X_train, Y_train)
predictions = clf.predict(X_test)
printResults(Y_test, predictions)

print("SVM-kernal=rbf")
clf=svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.1)
training_error = clf.fit(X_train,Y_train).score(X_train, Y_train)
predictions = clf.predict(X_test)
printResults(Y_test, predictions)






