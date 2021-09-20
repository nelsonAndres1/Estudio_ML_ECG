# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:47:53 2021

@author: andres agreda
"""
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #holdout
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import numpy as np
import matplotlib.pylab as plt 

data = np.loadtxt('datosFinal/datos.dat')
target = np.loadtxt('datosFinal/etiquetas.dat')

x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size = 0.2)# divide los datos de entrenamiento y prueba de modelo

print("Resultados Datos ( Filtrado y Luego Segmentación KNN)")
#print("Resultados Datos ( segmentación y Luego filtrado KNN)")


classifier=KNeighborsClassifier(n_neighbors=7, metric='manhattan') #creación clasificador
classifier.fit(x_train, y_train) #training
print(classifier.score(x_test, y_test))
predict_label=classifier.predict(x_test)
fpr, tpr, _ = roc_curve(y_test, predict_label)

plt.plot(fpr, tpr, lw=2, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve KNN')
plt.figure()
plt.show()  



MC=confusion_matrix(y_test, predict_label)
# Confusion Matrix
print("Confusion Matrix ")
print(MC)
# Accuracy
print("Accuracy ",accuracy_score(y_test, predict_label))

# Recall o sensibilidad 
print("Recall ",recall_score(y_test, predict_label, average=None))

# Precision
print("Precision ",precision_score(y_test, predict_label, average=None))

#Specificity
SP=(MC[1][1])/((MC[1][1])+(MC[0][1]))

print("Specificity ",SP)


