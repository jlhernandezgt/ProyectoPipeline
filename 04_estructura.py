# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:57:41 2022

@author: luish
"""
import funciones_proyecto as fn
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import resample
import time
from sklearn.utils import resample #para bootstraping.
from sklearn.linear_model import LogisticRegression


#os.chdir('/Users/luish/Documents/Maestria/trimestre2/StatisticalLearning/proyecto2/Proyecto2')

data_o = pd.read_csv("churn.csv")
data = pd.DataFrame(data_o[0:1000])

data.head()


### numericos ####
col_numerics = fn.getContinuesCols(data)
col_categoricals = fn.select_categorical_cols(data)


data[col_numerics].isnull().mean()

numeric_data = data[col_numerics]
categorical_data = data[col_categoricals]

numeric_data[col_numerics].isnull().mean()

numeric_data.head()

for col in col_numerics:
    fn.plot_density_variable(numeric_data, col) 

fn.FillNaN_Corr_DF(numeric_data, 'points_in_wallet', 'age' )
numeric_data[col_numerics].isnull().mean()

dataset_final = fn.funcion_final(numeric_data, 'avg_transaction_value','age', 1.75)

############ ------------------- 0 ---------------###################




########  CATEGORICOS    ##############  
fn.balanceo_datos(data, 'gender')    
data_balanceada = data  
fn.FillNaN_Corr_DF(data_balanceada, 'points_in_wallet', 'age' )
#data_balanceada[data_balanceada['gender'] == 'Unknown'] = 'M'

fn.balanceo_datos(data_balanceada, 'gender')  



### funcion de reproceso de data  ---- por confirmar
#proceso de balanceo de data.
#nMasculino = len(data_balanceada[data_balanceada['gender'] == "M"])
#Masculino = data_balanceada[data_balanceada['gender'] == "M"]
#Femenino = data_balanceada[data_balanceada['gender'] == "F"]
#Femenino = Femenino.sample(nMasculino, random_state=1234, replace=True)
#data_balanceada = Femenino.append(Masculino)
data_balanceada = data_balanceada.sample(frac=1, random_state=1234)
data_balanceada




#fn.FillNaN_Corr_DF(data_balanceada, 'points_in_wallet', 'age' )
#X = data_balanceada[['avg_transaction_value', 'points_in_wallet']]
X = data_balanceada[['avg_transaction_value','points_in_wallet']]
y = data_balanceada['gender']

#Ingeniería de caracteristicas - Codificación del Target.
lableEncoder = LabelEncoder()
lableEncoder.fit(['M', 'F'])
y = lableEncoder.transform(y.values)



#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle=True, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle=True, random_state=1234)

#modelo = SVC(C = 1, kernel = 'linear', random_state=123)
#modelo.fit(X_train, y_train)

start = time.time()
y_preds_svm = fn.modelo_svm(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_svm))
     
conf_matrix = pd.crosstab(y_test, y_preds_svm, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_svm(conf_matrix)




start = time.time()
y_preds_nb = fn.modelo_naive_bayes(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")


print("Accuracy: ", accuracy_score(y_test, y_preds_nb))

conf_matrix = pd.crosstab(y_test, y_preds_nb, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)    

fn.validacion_nb(conf_matrix)




start = time.time()
y_preds_tree = fn.modelo_arbol_decision(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_tree))

conf_matrix = pd.crosstab(y_test, y_preds_tree, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_dt(conf_matrix)



start = time.time()
y_preds_knn = fn.modelo_knn(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_knn))

conf_matrix = pd.crosstab(y_test, y_preds_knn, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_knn(conf_matrix)




start = time.time()
y_preds_lda = fn.modelo_lda(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")


print("Accuracy: ", accuracy_score(y_test, y_preds_lda))

conf_matrix = pd.crosstab(y_test, y_preds_lda, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_lda(conf_matrix)




start = time.time()
y_preds_qda = fn.modelo_qda(X_train, y_train, X_test)
end = time.time()
print("tiempo de entrenamiento: ", round(end - start), "segundos.")

print("Accuracy: ", accuracy_score(y_test, y_preds_qda))

conf_matrix = pd.crosstab(y_test, y_preds_qda, rownames=["observación"], colnames=["Predicción"])
print("Matriz de Confusión: \n\n", conf_matrix)

fn.validacion_qda(conf_matrix)



fn.print_roc(y_test,y_preds_svm,y_preds_nb,y_preds_tree,y_preds_knn,y_preds_lda, y_preds_qda)


svm_prob, svm_prob_v, _ = roc_curve(y_test, y_preds_svm)
nb_prob, nb_prob_v, _ = roc_curve(y_test, y_preds_nb)
tree_prob, tree_prob_v, _ = roc_curve(y_test, y_preds_tree)
knn_prob, knn_prob_v, _ = roc_curve(y_test, y_preds_knn)
lda_prob, lda_prob_v, _ = roc_curve(y_test, y_preds_lda)
qda_prob, qda_prob_v, _ = roc_curve(y_test, y_preds_qda)

plt.plot(svm_prob, svm_prob_v, linestyle="--", label="SVM")
plt.plot(nb_prob, nb_prob_v, marker='.', label="NB")
plt.plot(tree_prob, tree_prob_v, marker='.', label="tree")
plt.plot(knn_prob, knn_prob_v, marker='.', label="KNN")
plt.plot(lda_prob, lda_prob_v, marker='.', label="LDA")
plt.plot(qda_prob, qda_prob_v, marker='.', label="QDA")
plt.title("ROC Plot")
plt.xlabel("False Posotive Rate")
plt.ylabel("True Posotive Rate")
plt.legend()
plt.show()





lr = 'LogisticRegression().fit(X_train, y_train)'
svm = 'SVC(gamma="scale", kernel="rbf").fit(X_train, y_train)'
dt = 'DecisionTreeClassifier(criterion="gini", max_depth=4).fit(X_train, y_train)'
knn = 'KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)'
nb = 'GaussianNB().fit(X_train, y_train)'
lda = 'LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)'
qda = 'QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)'

models_list = {"lr" :lr, "svm":svm, "dt": dt,  "knn":knn, "nb":nb, "lda":lda, "qda":qda}
models_list



y_train = StandardScaler().fit_transform(y_train)

y_train = pd.DataFrame(y_train)
y_train.columns = ['gender']

dataset = pd.concat([X_train, y_train], axis=1)
dataset
fn.FillNaN_Corr_DF(dataset, 'points_in_wallet', 'gender' )
fn.FillNaN_Corr_DF(dataset, 'avg_transaction_value', 'gender' )
fn.FillNaN_Corr_DF(dataset, 'gender', 'gender' )

train_models_list = {}

for model_name, model in models_list.items():
    boot = resample(dataset, replace=True, n_samples=200, random_state=2020)
    X_train = boot.drop('gender', axis = 1)
    y_train = boot.gender
    train_model = eval(model)
    train_models_list[model_name] = train_model
    
train_models_list


auc_scores = {}
results_matrix = pd.DataFrame(columns=np.arange(0, len(y_test), 1).tolist())

for model_name, train_model in train_models_list.items():
    predicciones = train_model.predict(X_test)
    auc = roc_auc_score(y_test, predicciones)
    auc_scores[model_name] = auc
    tempDf = pd.DataFrame(predicciones).T
    results_matrix = results_matrix.append(tempDf)
    
    
    
results_matrix.index=list(train_models_list.keys())

results_matrix


votacion = results_matrix.apply(pd.value_counts)
votacion



final_predictions = []

for (columnName, columnData) in votacion.iteritems():
    column_result = columnData.values
    final_predictions.append(np.nanargmax(column_result, axis=0))

final_predictions

auc = roc_auc_score(y_test, final_predictions)
auc

auc_scores

