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

ICU_df = pd.read_csv('ICU_df.csv')

print("ICU dead: " + str(ICU_df['hospital_expire_flag'].sum()))
print("ICU ALive: " + str(ICU_df.shape[0] - ICU_df['hospital_expire_flag'].sum() ))

y = ICU_df.hospital_expire_flag# define the target variable (dependent variable) as y

#drop the hospital expire flag from the data 
#ICU_df.drop(['hospital_expire_flag', 'index', 'index.1', 'level_0', 'Unnamed: 0', 'lactate_min_labs'], axis=1, inplace=True)
# create training and testing vars, 
X_train, X_test, Y_train, Y_test = train_test_split(ICU_df, y, test_size=0.2, random_state=32)


corr = ICU_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(ICU_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(ICU_df.columns)
ax.set_yticklabels(ICU_df.columns)
sol = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
print(sol)
print('ihrevbv')
variablesToRemove = (sol.loc[:,(sol > 0.9)])
print(variablesToRemove.index.values[:])
var = []
for x in variablesToRemove.index.values:
	var.append(x[0])
print(var)

ICU_df.drop(var, axis=1, inplace=True)
print(list(ICU_df))

ICU_df.to_csv("ICU_df_removed.csv")
plt.show()