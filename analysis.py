#full analysis
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

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(X_train.columns)


matplotlib.rcParams.update({'font.size': 15})

# ICU_df = pd.read_csv('ICU_df.csv')

# print("ICU dead: " + str(ICU_df['hospital_expire_flag'].sum()))
# print("ICU ALive: " + str(ICU_df.shape[0] - ICU_df['hospital_expire_flag'].sum() ))

# y = ICU_df.hospital_expire_flag# define the target variable (dependent variable) as y

# #drop the hospital expire flag from the data
# ICU_df.drop(['hospital_expire_flag', 'index','index.1', 'level_0', 'Unnamed: 0', 'lactate_min_labs'], axis=1, inplace=True)
# # create training and testing vars,
# #ICU_df.drop(['hospital_expire_flag', 'level_0', 'Unnamed: 0'], axis=1, inplace=True)
# X_train, X_test, Y_train, Y_test = train_test_split(ICU_df, y, test_size=0.2, random_state=32)



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

print("\n\nRandom  Forest: ")
rf = RandomForestClassifier(n_estimators = 100, max_depth=6, n_jobs=4, oob_score=True)
print(rf.get_params())
training_error = rf.fit(X_train, Y_train).score(X_train, Y_train)
threshold = 0.4

predictions = rf.predict(X_test)
#predictions = (predicted_proba [:] >= threshold).astype('int')
print("\ntraining error: " + str(training_error))
print("\ntraining error: " + str(training_error))
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(list(X_train), importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('{:20} : {}'.format(*pair)) for pair in feature_importances];


probs = rf.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
aucList.append(auc)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='RF')
# show the plot
pyplot.legend()



# importances = rf.feature_importances_
# std = np.std([rf.feature_importances_ for tree in rf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# Print the feature ranking
# print("Feature ranking:")

# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), rf.feature_importances_,
#        color="r", align="center", )
# plt.xticks(range(X_train.shape[1]), list(X_train))
# plt.xlim([-1, X_train.shape[1]])





print("\n\nNeural net: ")
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(9, 7))
training_error = mlp.fit(X_train,Y_train).score(X_train, Y_train)
predictions = mlp.predict(X_test)

print("\ntraining error: " + str(training_error))
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))

probs = mlp.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
aucList.append(auc)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='NN')
# show the plot

print("SVM")
clf=svm.SVC(kernel='linear', probability=True, C=1, gamma=1e-05)
training_error = clf.fit(X_train,Y_train).score(X_train, Y_train)
predictions = clf.predict(X_test)
print("\ntraining error: " + str(training_error))
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))

probs = clf.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
aucList.append(auc)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='SVM-lin')

print("SVM-kernal=rbf")
clf=svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.1)
training_error = clf.fit(X_train,Y_train).score(X_train, Y_train)
predictions = clf.predict(X_test)
print("\ntraining error: " + str(training_error))
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))

probs = clf.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
aucList.append(auc)
print('AUC: %.3f' % auc)
print(confusion_matrix(Y_test, predictions)[0][1])
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='SVM-rfb')

pyplot.legend()
pyplot.show()



