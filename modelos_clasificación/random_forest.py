# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:18:34 2021

@author: Andres Agreda
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split #holdout
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

import numpy as np
import matplotlib.pylab as plt 


"""
data = np.loadtxt('DatosSF\matriz_datos2.dat')
target = np.loadtxt('DatosSF\matriz_etiquetas2.dat')

"""

data = np.loadtxt('DatosBrutos\dtsComProm.dat')
target = np.loadtxt('DatosBrutos\etqComProm.dat')


#print("Resultados Datos ( Filtrado y Luego Segmentación Random Forest)")
print("Resultados Datos ( segmentación y Luego filtrado Random Forest)")
"""
data = np.loadtxt('DatosSFWD\datosSFW.dat')
target = np.loadtxt('DatosSFWD\etiquetasSFW.dat')
"""

x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size = 0.2)# d
classifier=RandomForestClassifier(n_estimators = 100, criterion='entropy', max_depth=2, random_state=0)


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
plt.title('ROC curve Random Forest')
plt.figure()
plt.show()  



MC=confusion_matrix(y_test, predict_label)
# Confusion Matrix
print("Confusion Matrix \n",MC)
# Accuracy
print("Accuracy ",accuracy_score(y_test, predict_label))

# Recall o sensibilidad 
print("Recall ",recall_score(y_test, predict_label, average=None))

# Precision
print("Precision ",precision_score(y_test, predict_label, average=None))

#Specificity
SP=(MC[1][1])/((MC[1][1])+(MC[0][1]))

print("Specificity ",SP)
