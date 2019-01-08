# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 07:42:08 2018

@author: Ossama Mahmoud and Selam Mequanint
Purpose: Data processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import neighbors
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import GridSearchCV


#importing the dataset
data_df=pd.read_csv('data6hours.csv')


#list of columns in the original dataset
columns=list(data_df)
#identifying columns with no values

#remove columns with over the threshold missing value
reduced_df=data_df.dropna(axis=1,how='all',thresh=18073)# at least 40% is non-missingmove records with over 10% missing
#list f columsn after the reduction
columns=list(reduced_df)
# print(columns)
misscount= reduced_df.isnull().sum(axis=1)
# print("List of reduced colomns to only ones with more than 40%")
misscount.describe()
#further reduced to exclue non-predictive columns and response features
reduced_featurs= reduced_df.drop(['icustay_id', 'icustay_seq', 'los_hospital', 'los_icu', 'oasis','oasis_prob', 'sapsii', 'sapsii_prob', 'intime', 'hadm_id','specimen'], axis=1)

#identify rows with count of missing
missvalue= reduced_featurs.isnull().sum(axis='columns')
missvalue.shape
print(missvalue.describe())
#Keep only the rows with at least 5 non-NA values from the reduced datast.
print('reduced features')
print(reduced_featurs.shape)
reduced_featurs = reduced_featurs.dropna(thresh=71)
print(reduced_featurs.shape)
print('reduced features')

#replacing missings with mean value
cat = reduced_featurs[['gender', 'intubated', 'hospital_expire_flag' ]].copy().reset_index(drop=True)
reduced_featurs.drop(['gender', 'intubated', 'hospital_expire_flag' ], axis=1, inplace=True)
reduced_featurs = reduced_featurs.reset_index(drop=True)

reduced_featurs.fillna(reduced_featurs.mean(), inplace=True)
cat.fillna(0, inplace=True)

reduced_featurs = pd.concat([reduced_featurs, cat], axis=1).reset_index()

# count the number of NaN values in each column


###Under-sampling
Deceased = len(reduced_featurs[reduced_featurs['hospital_expire_flag'] == 1])
print("\nDeceased:   %d" %Deceased)
print("\nDishcharged:   %d" %len(reduced_featurs[reduced_featurs['hospital_expire_flag'] == 0]))

Discharged_indices = reduced_featurs[reduced_featurs['hospital_expire_flag'] == 0].index
random_indices = np.random.choice(Discharged_indices,Deceased, replace=False)
Deceased_indices = reduced_featurs[reduced_featurs.hospital_expire_flag == 1].index
under_sample_indices = np.concatenate([Deceased_indices,random_indices])
under_sample = reduced_featurs.loc[under_sample_indices]

print("\nUnder sampled Deceased:   %d" %len(under_sample[under_sample['hospital_expire_flag'] == 1]))
print("\nUnder sampled Dishcharged:   %d" %len(under_sample[under_sample['hospital_expire_flag'] == 0]))



ICU_df = under_sample

ICU_df.rename(index=str, columns={'sodium_max.1': 'sodium_max_labs', 'sodium_min.1': 'sodium_min_labs','potassium_min.1': 'potassium_min_labs', 'potassium_max.1': 'potassium_max_labs', 'lactate_min.1':'lactate_min_labs', 'lactate_max.1':'lactate_max_labs', 'hemoglobin_min.1':'hemoglobin_min_labs', 'hemoglobin_max.1':'hemoglobin_max_labs',  'hematocrit_min.1':'hematocrit_min_labs', 'hematocrit_max.1':'hematocrit_max_labs', 'chloride_min.1':'chloride_min_labs',  'chloride_max.1':'chloride_max_labs',  'bicarbonate_min.1':'bicarbonate_min_labs', 'bicarbonate_max.1':'bicarbonate_max_labs', 'bicarbonate_min.1':'bicarbonate_min_labs'}, inplace = True)


ICU_df.drop(columns=['glucose_min.2', 'glucose_max.2', 'subject_id'], inplace=True)


#calculating BMI
cm_to_meter_ratio = 0.01
ICU_df['height_meter'] = cm_to_meter_ratio*ICU_df['height_first']
ICU_df['BMI'] = ICU_df['weight_first']/(ICU_df['height_meter']**2)

#new variable of respratemean
ICU_df['respxvent'] = ICU_df['intubated'] * ICU_df['resprate_mean']




scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(ICU_df.drop(['gender', 'intubated', 'hospital_expire_flag' ], axis=1))
#readd gender and intubated
scaled_df = pd.DataFrame(scaled_df, columns=list(ICU_df.drop(['gender', 'intubated', 'hospital_expire_flag' ], axis=1)))


df_cat = ICU_df[['gender', 'intubated', 'hospital_expire_flag' ]].copy().reset_index()
scaled_df = scaled_df.reset_index()
ICU_df = pd.concat([scaled_df, df_cat], axis=1)

ICU_df['gender'] = ICU_df['gender'].replace('F',0)
ICU_df['gender'] = ICU_df['gender'].replace('M',1)


ICU_df.to_csv("ICU_df.csv")
print(ICU_df)

print(ICU_df.shape)

y = ICU_df.hospital_expire_flag# define the target variable (dependent variable) as y

#drop the hospital expire flag from the data and the indexes
print(ICU_df.columns)
ICU_df.drop(['hospital_expire_flag', 'level_0', 'index' ], axis=1, inplace=True)

# create training and testing vars
X_train, X_test, Y_train, Y_test = train_test_split(ICU_df, y, test_size=0.2, random_state=43)

X_train['gender'].replace('F',0,inplace=True)
X_train['gender'].replace('M',1,inplace=True)

X_test['gender'].replace('F',0,inplace=True)
X_test['gender'].replace('M',1,inplace=True)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
X_train.to_csv('X_train.csv', index=False)
Y_train.to_csv('y_train.csv', header=['Y'])
X_test.to_csv('X_test.csv', index=False)
Y_test.to_csv('y_test.csv', header=['Y'])




# X_train=pd.read_csv('X_train.csv')
# Y_train=pd.read_csv('y_train.csv')
# print("Xtrain:  "+ str(X_train.shape))
# print("Ytrain:  "+ str(Y_train.shape))
# # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
# # 	print(X_train)
# print("Ytrain:  "+ str(Y_train.shape))
# X_test=pd.read_csv('X_test.csv')
# Y_test=pd.read_csv('Y_test.csv')



# print(list(X_train))




print("expired: " + str(Y_train.sum()))
print("Alive: " + str(Y_train.size - Y_train.sum() ))
