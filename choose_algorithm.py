# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:40:57 2020

@author: 747
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import  metrics 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
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
os.listdir(r'C:\Users\747\Desktop\暑研')
brainwave_df = pd.read_csv(r'C:\Users\747\Desktop\暑研/emotions.csv')
label_df = brainwave_df['label']
brainwave_df.drop('label', axis = 1, inplace=True)


pl_random_forest = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(loss='l2', penalty='l1', dual=False))),
  ('classification', RandomForestClassifier(n_estimators = 90,oob_score = True,n_jobs = -1,
  random_state =50))
])

pl_random_forest.fit(brainwave_df,label_df)
scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())
# min_samples_split=80,  min_samples_leaf=20
# max_depth=11,9; max_features=9

'''
pl_random_forest.fit(brainwave_df,label_df)
scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())
#save
#joblib.dump(pl_random_forest,'save/test01.pkl')
print('success!')
'''
'''
param_test1= {'n_estimators':range(70,101,10)}  
gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,  
                                 min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=50),  
                       param_grid =param_test1,cv=5)  
gsearch1.fit(brainwave_df,label_df)  
gsearch1.cv_results_,gsearch1.best_params_, gsearch1.best_score_ 
'''
'''
param_test2= {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}  
gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators=90,  
                                 min_samples_leaf=20, max_features='sqrt',oob_score=True ,random_state=50),  
                       param_grid =param_test2, cv=10)  
gsearch2.fit(brainwave_df,label_df)  
gsearch2.best_score_ 

#点gsearch双击best_param
'''

'''
param_test3= {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}  
gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90,max_depth=9,  
                                 max_features='sqrt' ,oob_score=True, random_state=50),  
   param_grid = param_test3,iid=False, cv=5)  
gsearch3.fit(brainwave_df,label_df)  
gsearch3.cv_results_,gsearch3.best_params_, gsearch3.best_score_ 
'''
'''
param_test4= {'max_features':range(3,11,2)}  
gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90,  
oob_score=True, random_state=50),  
   param_grid = param_test4,iid=False, cv=5)  
gsearch4.fit(brainwave_df,label_df)  
gsearch4.cv_results_,gsearch4.best_params_, gsearch4.best_score_
'''
endtime = datetime.datetime.now()
print (endtime - starttime)
