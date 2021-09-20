# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:35:20 2021

@author: Andres Agreda
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split #holdout
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


import numpy as np
import matplotlib.pylab as plt 


"""
data = np.loadtxt('DatosSF\matriz_datos2.dat')
target = np.loadtxt('DatosSF\matriz_etiquetas2.dat')

data = np.loadtxt('DatosFS\matriz_Dates.dat')
target = np.loadtxt('DatosFS\matriz_etiquetas.dat')


data = np.loadtxt('DatosFSWD\datosFFW.dat')
target = np.loadtxt('DatosFSWD\etiquetasFFW.dat')


data = np.loadtxt('DatosSFWD\datosSFW.dat')
target = np.loadtxt('DatosSFWD\etiquetasSFW.dat')
"""

data = np.loadtxt('DatosBrutos\dtsComProm.dat')
target = np.loadtxt('DatosBrutos\etqComProm.dat')

print("Resultados Datos ( segmentación y Luego filtrado Redes Neuronales) Y Wavelet")

x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size = 0.2)# d


classifier = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', 
                           alpha=0.0001, batch_size='auto', learning_rate='constant',
                           learning_rate_init=0.001, power_t=0.5, max_iter=400, 
                           shuffle=True, random_state=None, tol=0.0001, verbose=False,
                           warm_start=False, momentum=0.9, nesterovs_momentum=True,
                           early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

classifier.fit(x_train, y_train)
print(classifier.score(x_test,y_test))


predict_label=classifier.predict(x_test)
fpr, tpr, _ = roc_curve(y_test, predict_label)

plt.plot(fpr, tpr, lw=2, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve RN')
plt.figure()
plt.show()  



MC=confusion_matrix(y_test, predict_label)
# Confusion Matrix
print("Confusion Matrix ",MC)
# Accuracy
print("Accuracy ",accuracy_score(y_test, predict_label))

# Recall o sensibilidad 
print("Recall ",recall_score(y_test, predict_label, average=None))

# Precision
print("Precision ",precision_score(y_test, predict_label, average=None))

#Specificity
SP=(MC[1][1])/((MC[1][1])+(MC[0][1]))

print("Specificity ",SP)
