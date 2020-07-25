# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:34:23 2020

@author: 747
"""
import numpy as np
import pandas as pd
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
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import os
os.listdir(r'C:\Users\747\Desktop\暑研')
brainwave_df = pd.read_csv(r'C:\Users\747\Desktop\暑研/emotions.csv')

label_df = brainwave_df['label']
brainwave_df.drop('label', axis = 1, inplace=True)

##%%time
pl_random_forest = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(loss='l2', penalty='l1', dual=False))),
  ('classification', RandomForestClassifier())
])

# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix


labels = ['positive', 'negative', 'neutron']



y_true = label_df
y_pred = pl_random_forest.fit(brainwave_df, label_df).predict(brainwave_df)
print(y_pred)
print(y_true)
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig(r'C:\Users\747\Desktop\暑研\result', format='png')
plt.show()
