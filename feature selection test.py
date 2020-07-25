# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:49:51 2020

@author: 747
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
import datetime
import os

from sklearn.externals import joblib
starttime = datetime.datetime.now()
#long running

os.listdir(r'C:\Users\747\Desktop\暑研')
brainwave_df = pd.read_csv(r'C:\Users\747\Desktop\暑研/emotions.csv')
label_df = brainwave_df['label']
brainwave_df.drop('label', axis = 1, inplace=True)

'''
pl_mlp = Pipeline(steps=[('scaler',StandardScaler()),
                             ('mlp_ann', MLPClassifier(hidden_layer_sizes=[1275,637],
                                    activation='tanh',alpha=1,random_state=50)

)])
scores = cross_val_score(pl_mlp, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for ANN : ', scores.mean())
'''

'''
pl_random_forest = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(loss='l2', penalty='l1', dual=False))),
  ('classification', RandomForestClassifier(n_estimators = 90,oob_score = True,n_jobs = -1,
  random_state =50))
])
pl_random_forest.fit(brainwave_df,label_df)
scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())

#save
joblib.dump(pl_random_forest,'save/pl_random_forest02.pkl')

#restore
clf=joblib.load('save/pl_random_forest02.pkl')

#print(clf.predict(brainwave_df[0:1]))
scores = cross_val_score(clf, brainwave_df, label_df,cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())
'''
clf=joblib.load('save/pl_random_forest02.pkl')
#print(clf.predict(brainwave_df[0:1]))
scores = cross_val_score(clf, brainwave_df, label_df,cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())
endtime = datetime.datetime.now()
print (endtime - starttime)
#Recall for RandomForest :  0.985894999627394
#scores = cross_val_score(clf, brainwave_df, label_df,cv=10,scoring='recall_macro')
