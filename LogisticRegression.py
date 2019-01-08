
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
import matplotlib


# X_train=pd.read_csv('X_train.csv')
# Y_train=pd.read_csv('y_train.csv')

# # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# # 	print(X_train)
# X_test=pd.read_csv('X_test.csv')
# Y_test=pd.read_csv('Y_test.csv', index=False)


# print(Y_test)

# exit()
# print(list(X_train))

ICU_df = pd.read_csv('ICU_df.csv')

print("ICU dead: " + str(ICU_df['hospital_expire_flag'].sum()))
print("ICU ALive: " + str(ICU_df.shape[0] - ICU_df['hospital_expire_flag'].sum() ))

y = ICU_df.hospital_expire_flag# define the target variable (dependent variable) as y

#drop the hospital expire flag from the data 
ICU_df.drop(['hospital_expire_flag', 'index','index.1', 'level_0', 'Unnamed: 0', 'lactate_min_labs'], axis=1, inplace=True)
# create training and testing vars, 
#ICU_df.drop(['hospital_expire_flag', 'level_0', 'Unnamed: 0'], axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(ICU_df, y, test_size=0.2, random_state=32)

print("\n\nRandom  Forest: ")
rf = RandomForestClassifier(n_estimators = 100, max_depth=6, random_state = 87, n_jobs=4, oob_score=True)
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

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(list(X_train), importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('{:20} : {}'.format(*pair)) for pair in feature_importances];

matplotlib.rcParams.update({'font.size': 24})

probs = rf.predict_proba(X_test)
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs)

# plot the roc curve for the model
pyplot.plot(fpr, tpr, label='RF')
# show the plot
pyplot.legend()

pyplot.show()


# mlp = MLPClassifier()
# # training_error = mlp.fit(X_train,Y_train).score(X_train, Y_train)
# # predictions = mlp.predict(X_test)

# hidden_layer_sizes = [(9,4), (9,5), (9,6), (9,7), (10,4),(10,5), (10,6), (10,7)]
# print(mlp.get_params().keys())
# parameters = { 	'hidden_layer_sizes' :hidden_layer_sizes,
#     			'activation': ["relu", "tanh"]}
# gridsearch = GridSearchCV(mlp, parameters, verbose=10)
# #best Params = {'activation': 'tanh', 'hidden_layer_sizes': (10, 5)}
# #gridsearch.fit(X_train, Y_train)
# #print("best Params = " + str(gridsearch.best_params_))

# mlp = MLPClassifier(activation= 'tanh', hidden_layer_sizes=(9, 7))
# training_error = mlp.fit(X_train,Y_train).score(X_train, Y_train)
# predictions = mlp.predict(X_test)
# print("\ntraining error: " + str(training_error))
# print(accuracy_score(Y_test, predictions))
# print(confusion_matrix(Y_test, predictions))
# print(classification_report(Y_test, predictions))

