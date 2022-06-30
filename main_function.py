# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:24:53 2022

@author: luish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score  
import time as tm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

def train_model(x,y):
    ejecucion = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    K_fold = 10
    tiempo = {}
    tiempo['Start Train Model'] = 'Fecha Ejecucion:', ejecucion,''
    scores = {}
    results = pd.DataFrame()

    """
        Modelo LDA
    """
    start = tm.time()
    LDA = LinearDiscriminantAnalysis()
    grid_search_lda = {"solver" : ["svd"], "tol" : [0.0001,0.0002,0.0003]}
    gsLDA = GridSearchCV(LDA, param_grid = grid_search_lda, cv=K_fold, scoring="accuracy", n_jobs= 4, verbose = 1)
    gsLDA.fit(x,y)
    scores['Modelo LDA'] = gsLDA.best_score_
    score_df = pd.DataFrame(gsLDA.cv_results_).assign(modelo='Modelo LDA')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)
    end = tm.time()
    tiempo['Modelo LDA'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'

    """
    Modelo Naive Bayes
    """
    start = tm.time()
    NB =  GaussianNB()
    grid_search_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
    gs_NB = GridSearchCV(estimator=NB, param_grid=grid_search_nb,  cv=K_fold,  verbose=1,  scoring='accuracy') 
    gs_NB.fit(x, y)
    end = tm.time()
    tiempo['Modelo NB'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'
    scores['Modelo NB'] = gs_NB.best_score_
    score_df = pd.DataFrame(gs_NB.cv_results_).assign(modelo='Modelo NB')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)


    """
    Modelo SVM
    """
    start = tm.time()
    SVMC = SVC(probability=True)
    grid_search_svm = {'kernel': ['rbf'],'gamma': [0.0001, 0.001, 0.01, 0.1, 1],'C': [1, 10, 50, 100]}
    gsSVMC = GridSearchCV(SVMC, param_grid = grid_search_svm, cv = K_fold, scoring="accuracy", n_jobs= -1, verbose = 1)
    gsSVMC.fit(x,y)
    end = tm.time()
    tiempo['Modelo SVM'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'
    scores['Modelo SVM'] = gsSVMC.best_score_
    score_df = pd.DataFrame(gsSVMC.cv_results_).assign(modelo='Modelo SVM')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)


    """
    Modelo Random Forest
    """
    start = tm.time()
    RFC = RandomForestClassifier()
    grid_search_rf = {"max_depth": [None],"min_samples_split": [2, 6, 20],"min_samples_leaf": [1, 4, 16],"n_estimators" :[100,200,300,400],"criterion": ["gini"]}
    gsRFC = GridSearchCV(RFC, param_grid = grid_search_rf, cv=K_fold, scoring="accuracy", n_jobs= 4, verbose = 1)
    gsRFC.fit(x,y)
    end = tm.time()
    tiempo['Modelo RFC'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'
    scores['Modelo RFC'] = gsRFC.best_score_
    score_df = pd.DataFrame(gsRFC.cv_results_).assign(modelo='Modelo RFC')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)

    """
    Modelo KNN
    """
    start = tm.time()
    knn = KNeighborsClassifier()
    grid_search_knn = { 'n_neighbors' : [5,7,9,11,13,15], 'weights' : ['uniform','distance'], 'metric' : ['minkowski','euclidean','manhattan']}
    gs_knn = GridSearchCV(KNeighborsClassifier(), grid_search_knn, verbose = 1, cv=K_fold, n_jobs = -1)
    gs_knn.fit(x, y)
    end = tm.time()
    tiempo['Modelo KNN'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'
    scores['Modelo KNN'] = gs_knn.best_score_
    score_df = pd.DataFrame(gs_knn.cv_results_).assign(modelo='Modelo KNN')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)

    """ 
    Modelo QDA
    """
    start = tm.time()
    DQA = QuadraticDiscriminantAnalysis()
    grid_search_qda = {
        'reg_param': (0.00001, 0.0001, 0.001,0.01, 0.1), 
        'store_covariance': (True, False),
        'tol': (0.0001, 0.001,0.01, 0.1), 
                    }
    gs_qda = GridSearchCV(estimator=DQA, param_grid=grid_search_qda, scoring = 'accuracy', n_jobs = -1, cv = K_fold)
    gs_qda.fit(x, y)
    end = tm.time()
    tiempo['Modelo QDA'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'
    scores['Modelo QDA'] = gs_qda.best_score_
    score_df = pd.DataFrame(gs_qda.cv_results_).assign(modelo='Modelo QDA')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)

    """
    Modelo DTC
    """
    start = tm.time()
    DT = DecisionTreeClassifier(random_state=2022)
    grid_search_dt = {'max_depth': [2, 3, 5, 10, 20],'min_samples_leaf': [5, 10, 20, 50, 100],'criterion': ["gini", "entropy"]}
    gs_DT = GridSearchCV(estimator=DT, param_grid=grid_search_dt, cv=K_fold, n_jobs=-1, verbose=1, scoring = "accuracy")
    gs_DT.fit(x, y)
    end = tm.time()
    tiempo['Modelo DT'] = 'Tiempo de entrenamiento:', round(end -start),'segundos'
    scores['Modelo DT'] = gs_DT.best_score_
    score_df = pd.DataFrame(gs_DT.cv_results_).assign(modelo='Modelo DT')
    score_df = score_df.loc[:0,['modelo', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results = results.append(score_df)
    
    fin_ejecucion = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tiempo['End Train Model'] = 'Fecha Finalizacion:', fin_ejecucion,''


    tiempo = pd.DataFrame(tiempo)
    tiempo = tiempo.transpose()
    
    return(tiempo, scores, results)



