# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:57:17 2022

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




## funcion para detecetar variables continuas
def getContinuesCols(df):
    colnames = df.columns
    numeric_continues_vars = []
    for col in colnames:
        unique_values =len (df[col].unique())
        if((df[col].dtype != 'object') and (unique_values > 30)):
            numeric_continues_vars.append(col)
    return numeric_continues_vars

def select_categorical_cols(df):
    colnames = df.columns
    categorical_vars = []
    for col in colnames:
        if(df[col].dtype == 'object'):
            categorical_vars.append(col)
    return categorical_vars



##  imputacion media y mediana

## calculo media y rellenado de valores nulos
def ImpMeanV(df,col1):
    mv = np.round(df[col1].mean(),0)
    return(df[col1].fillna(mv))

## calculo mediana y rellenado de valores nulos
def ImpMedianV(df,col1):
    mv = np.round(df[col1].median(),0)
    return(df[col1].fillna(mv))

##  graficado de densisdad
def plt_dns_df(df,col1):
    df_mean = ImpMeanV(df, col1)
    df_median = ImpMedianV(df, col1)
    fig = plt.figure()
    fig.add_subplot(111)
    df[col1].plot.density(color = 'red')
    df_mean.plot.density(color = 'blue')
    df_median.plot.density(color = 'green')

## correlacion y fill de valores nulos
def FillNaN_Corr_DF(df, col1, col2):
    mean_val = np.round(df[col1].mean(), 0)
    print(f'La media es: {mean_val}')
    median_val = np.round(df[col1].median(), 0)
    print(f'La mediana es: {median_val}')
    df_LF_meanImp = df[col1].fillna(mean_val)
    df_LF_meadianImp = df[col1].fillna(median_val)
    corr1 = np.corrcoef(df_LF_meanImp, df[col2])[0,1]
    corr2 = np.corrcoef(df_LF_meadianImp, df[col2])[0,1]
    print(corr1)
    print(corr2)
    if corr1 >= corr2:
        df[col1] = df[col1].fillna(mean_val)
    else:
        df[col1] = df[col1].fillna(median_val)
    print('Validacion Valores Nulos:')
    print(df[col1].isnull().sum())
    
    
## funcion para graficar la variable con su densidad
def plot_density_variable(df, col1):
    
    plt.figure(figsize = (15,6))
    plt.subplot(121)
    df[col1].hist(bins=30)
    plt.title(col1)
    
    plt.subplot(122)
    stats.probplot(df[col1], dist="norm", plot=plt)
    plt.show()

## seleccion de nuevo df 
def new_df_trans(df, col1, col2):
    df = df.loc[:, [col1, col2]]
    return(df)


## transformacion metodo YeoJhonson
def trans_YeoJohnson(df, col1, col2):
    df[col1+"_YJ"], lambdaX = stats.yeojohnson(df[col2])
    print("correlacion: ", np.corrcoef(df[col1+"_YJ"], df[col2])[0, 1])
    plot_density_variable(df, col1+"_YJ")
    return(df)

#Outliers
def inspect_outliers(df, col1):
    
    plt.figure(figsize = (15,6))
    
    plt.subplot(131)
    sns.distplot(df[col1], bins=30)
    plt.title("Densisd-Histograma: " + col1)
    
    plt.subplot(132)
    stats.probplot(df[col1], dist="norm", plot=plt)
    plt.title("QQ-Plot: " + col1)
    
    plt.subplot(133)
    sns.boxplot(y=df[col1])
    plt.title("Boxplot: " + col1)
    
    plt.show()
    
##Función para detectar outliers
def detect_outliers(df, col1, factor):
    IQR = df[col1].quantile(0.75) - df[col1].quantile(0.25)
    LI = df[col1].quantile(0.25) - (IQR*factor)
    LS = df[col1].quantile(0.75) + (IQR*factor)
    
    return LI, LS


## tratamiento de outliers
def outlier_treatment(df, col1, factor):
    IQR = df[col1].quantile(0.75) - df[col1].quantile(0.25)
    LI = df[col1].quantile(0.25) - (IQR*factor)
    LS = df[col1].quantile(0.75) + (IQR*factor)
    
    df[col1] = np.where(df[col1] > LS, LS,
                                          np.where(df[col1] < LI, LI, df[col1]))
    return(df)



## FeatureScaling
def FeatureScaling(df):
    scaler = StandardScaler()
    scaler.fit(df) #calcular parámetros de configuración para cada columna.
    StandardScaler()
    df_scaler = pd.DataFrame(scaler.transform(df), columns=df.columns)    
    return(df_scaler)


def funcion_final (df, col1, col2, factor):
    ImpMeanV(df, col1)
    ImpMedianV(df, col1)
    plt_dns_df(df, col1)
    FillNaN_Corr_DF(df, col1, col2)
    numeric_cont_vars = getContinuesCols(df)
    plot_density_variable(df, col1)
    
    for col in numeric_cont_vars:
            plot_density_variable(df, col)
            
    new_df = new_df_trans(df, col1, col2)
    plot_density_variable(new_df, col1)
    new_df = trans_YeoJohnson(new_df,col1,col2)   
    
    for col in numeric_cont_vars:
        inspect_outliers(df, col)
    
    detect_outliers(df, col1, factor)
    
    for col in numeric_cont_vars:
        outlier_treatment(df, col, factor)
    
    dataset_temp = df.loc[:, numeric_cont_vars]
    dataset_temp.describe()
    
    final = FeatureScaling(dataset_temp)
    return final



#verificación de balanceo de datos.
def balanceo_datos(df, col1):
    return(df[col1].value_counts())



#### modelo svc

def modelo_svm(Xtrain_df,ytrain_df,Xtest_df):
    svm = SVC(kernel="linear", C=1)
    svm.fit(Xtrain_df, ytrain_df)
    y_preds_svm = svm.predict(Xtest_df)
    return(y_preds_svm)


def validacion_svm(df):
    TP = df.iloc[1,1]
    TN = df.iloc[0,0]
    FN = df.iloc[1,0]
    FP = df.iloc[0,1]
    print("Sentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))




def modelo_naive_bayes(Xtrain_df,ytrain_df,Xtest_df):
    clf_nb = GaussianNB()
    clf_nb.fit(Xtrain_df, ytrain_df)
    y_preds_nb = clf_nb.predict(Xtest_df)
    return(y_preds_nb)

def validacion_nb(df):
    TP = df.iloc[1,1]
    TN = df.iloc[0,0]
    FN = df.iloc[1,0]
    FP = df.iloc[0,1]
    print("Sentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))




def modelo_arbol_decision(Xtrain_df,ytrain_df,Xtest_df):
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(Xtrain_df, ytrain_df)
    y_preds_tree = clf_tree.predict(Xtest_df)
    return(y_preds_tree)


def validacion_dt(df):
    TP = df.iloc[1,1]
    TN = df.iloc[0,0]
    FN = df.iloc[1,0]
    FP = df.iloc[0,1]
    print("Sentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))


def modelo_knn(Xtrain_df,ytrain_df,Xtest_df):
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(Xtrain_df, ytrain_df)
    y_preds_knn = clf_knn.predict(Xtest_df)
    return(y_preds_knn)


def validacion_knn(df):
    TP = df.iloc[1,1]
    TN = df.iloc[0,0]
    FN = df.iloc[1,0]
    FP = df.iloc[0,1]
    print("Sentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))
    
    


    
def modelo_lda(Xtrain_df,ytrain_df,Xtest_df):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda.fit(Xtrain_df, ytrain_df)
    y_preds_lda = lda.predict(Xtest_df)
    return(y_preds_lda)

def validacion_lda(df):
    TP = df.iloc[1,1]
    TN = df.iloc[0,0]
    FN = df.iloc[1,0]
    FP = df.iloc[0,1]
    print("Sentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))
    

def modelo_qda(Xtrain_df,ytrain_df,Xtest_df):
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(Xtrain_df, ytrain_df)
    y_preds_qda = qda.predict(Xtest_df)
    return(y_preds_qda)
    
def validacion_qda(df):
    TP = df.iloc[1,1]
    TN = df.iloc[0,0]
    FN = df.iloc[1,0]
    FP = df.iloc[0,1]
    print("Sentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))
    
def print_roc(ytest_df,y_svm,y_nb,y_tree,y_knn,y_lda,y_qda):
    print('ROC-ACU -> SVM = ', roc_auc_score(ytest_df, y_svm))
    print('ROC-ACU -> NB = ', roc_auc_score(ytest_df, y_nb))
    print('ROC-ACU -> Tree = ', roc_auc_score(ytest_df, y_tree))
    print('ROC-ACU -> KNN = ', roc_auc_score(ytest_df, y_knn))
    print('ROC-ACU -> LDA = ', roc_auc_score(ytest_df, y_lda))
    print('ROC-ACU -> QDA = ', roc_auc_score(ytest_df, y_qda))



"""
insertar funcion final para analisis categorico
"""



### funciones para pipeline

def getColumnsDataTypes(df):
    categoric_vars = []
    discrete_vars = []
    continues_vars = []

    for colname in df.columns:
        if(df[colname].dtype == 'object'):
            categoric_vars.append(colname)
        else:
            cantidad_valores = len(df[colname].value_counts())
            if(cantidad_valores <= 30):
                discrete_vars.append(colname)
            else:
                continues_vars.append(colname)

    return categoric_vars, discrete_vars, continues_vars




def plotCategoricalVals(df, categoric_vars, y):
    for column in categoric_vars:
        plt.figure(figsize=(12,6))
        plot = sns.countplot(x=df[column], hue=df[y])
        plt.show()
