#roc-curve.py
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn import svm
import matplotlib




def calculateROC(Y_test, probs):
	probs = probs[:, 1]
	# calculate AUC
	auc = roc_auc_score(Y_test, probs)

	# calculate roc curve
	fpr, tpr, thresholds = roc_curve(Y_test, probs)
	return (auc, fpr, tpr)


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
i = 0
# for x in LR.coef_[0]:
# 	print("%s :  %f "%( (list(X_train)[i]) , x))
# 	i+=1

# print("\ntraining error: " + str(training_error))
print("Accuracy: " +str(accuracy_score(Y_test, predictions)))
# print(confusion_matrix(Y_test, predictions))
# print(classification_report(Y_test, predictions))
# print("False -ve = " + str(confusion_matrix(Y_test, predictions)[1][0] / (confusion_matrix(Y_test, predictions)[1][0]+confusion_matrix(Y_test, predictions)[0][0])))


probs = LR.predict_proba(X_test)
(auc, fpr, tpr) = calculateROC(Y_test, probs)

print('AUC: %.3f' % auc)
# calculate roc curve

# plot control
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='LR')
# show the plot


print("\n\nDecision Tree: ")
DT = DecisionTreeClassifier(max_depth=6)
training_error = DT.fit(X_train, Y_train).score(X_train, Y_train)
predictions = DT.predict(X_test)

print("Accuracy: " +str(accuracy_score(Y_test, predictions)))



probs = DT.predict_proba(X_test)
(auc, fpr, tpr) = calculateROC(Y_test, probs)

print('AUC: %.3f' % auc)
# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='DT')
# show the plot

print("\n\nRandom  Forest: ")
rf = RandomForestClassifier(n_estimators = 100, max_depth=6, n_jobs=4, oob_score=True)
training_error = rf.fit(X_train, Y_train).score(X_train, Y_train)
threshold = 0.4

predictions = rf.predict(X_test)

print("Accuracy: " +str(accuracy_score(Y_test, predictions)))


probs = rf.predict_proba(X_test)
(auc, fpr, tpr) = calculateROC(Y_test, probs)

print('AUC: %.3f' % auc)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='RF')
# show the plot
pyplot.legend()








print("\n\nNeural net: ")
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(9, 7))
training_error = mlp.fit(X_train,Y_train).score(X_train, Y_train)
predictions = mlp.predict(X_test)


print("Accuracy: " + str(accuracy_score(Y_test, predictions)))

probs = mlp.predict_proba(X_test)
(auc, fpr, tpr) = calculateROC(Y_test, probs)

print('AUC: %.3f' % auc)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='NN')
# show the plot

print("SVM")
clf=svm.SVC(kernel='linear', probability=True, C=1, gamma=1e-05)
training_error = clf.fit(X_train,Y_train).score(X_train, Y_train)
predictions = clf.predict(X_test)
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))

probs = clf.predict_proba(X_test)
(auc, fpr, tpr) = calculateROC(Y_test, probs)

print('AUC: %.3f' % auc)
# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='SVM-lin')

print("SVM-kernal=rbf")
clf=svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.1)
training_error = clf.fit(X_train,Y_train).score(X_train, Y_train)
predictions = clf.predict(X_test)
print("Accuracy: " +str(accuracy_score(Y_test, predictions)))


probs = clf.predict_proba(X_test)
(auc, fpr, tpr) = calculateROC(Y_test, probs)

print('AUC: %.3f' % auc)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='SVM-rfb')

pyplot.legend()
pyplot.show()

