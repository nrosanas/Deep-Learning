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

train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
train_df.describe(include=['O'])
train_df.describe(include='all')

#Completar
#
# podem mirar les dades amb les ultimes comandes i veiem que hi ha
# categories que hem de completar com 'age' perquè segurament és important
# per predir la probabilitat de sobreviure. Potser també ho haurem de fer amb 
# la variable embarked.
#
# Corregir
#
# la feature ticket no ens diu massa re tenim moltes repetides, la eleiminarem
# Cabin també li falten moltes dades i no seria massa útil
# passanger Id no ens diu res i també l'extraurem
# Name no és massa standard i de moment la deixem tot i que potser no és massa
#rellevant
#
#Crear
#
#podem intentar fer una categoria amb el nombre de familiars a partir de sibsp
# i parch 
#podem buscar la manera de treure info del nom de la gent per veure si és útil
#per ajudar-nos a fer la classificació podriem convertir la categoria edat de 
# numerica continua a franges
#potser crear la categoria fare ens ajuda


#comprovem si el fet de pertanyer a diferenta categoria afecta
print('comprovem si el fet de pertanyer a diferenta categoria afecta')
print(train_df[['Pclass','Survived']].groupby(by='Pclass').mean())

#comprovem si el sexe afecta
print('comprovem si el sexe afecta')
print(train_df[['Sex','Survived']].groupby(by='Sex').mean())
