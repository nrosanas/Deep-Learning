# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:55:08 2017

@author: Noè
"""

import pandas as pd
import numpy as np
import random as rd 

import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# carreguem les dades

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

#mirem les etiquetes de cada feature
print(train_df.columns.values)
