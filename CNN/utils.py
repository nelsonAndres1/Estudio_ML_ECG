# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:15:18 2020

@author: JARS
"""
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt  

def plotting_save(history, true_label, predict_label, model_name, stage, times):
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1,len(acc)+1,1)
    fpr, tpr, _ = roc_curve(true_label, predict_label)

    plt.plot(epochs,     acc, 'r--', label='Training acc'  )
    plt.plot(epochs, val_acc,  'b', label='Validation acc')
    plt.title ('Training and validation accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend(['train acc','validation acc'])
    plt.figure()
    
    plt.plot( epochs,     loss, 'r--' )
    plt.plot( epochs, val_loss ,  'b' )
    plt.title ('Training and validation loss'   )
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend(['train loss','validation loss'])
    plt.figure()

    plt.plot(fpr, tpr, lw=2, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.figure()
    plt.show()  
    
       
    #####Printing
    print('AUC value', roc_auc_score(true_label, predict_label))
    print('Confusion Matrix\n', confusion_matrix(true_label,predict_label))
    print('F1-score', f1_score(true_label, predict_label))

    
    
    #####Saving
    model_data=np.vstack([['ACC','VAL_ACC','LOSS','VAL_LOSS'],
                    np.hstack([np.reshape(np.round(acc,3), [len(acc),1]), 
                    np.reshape(np.round(val_acc,3), [len(acc),1]), 
                    np.reshape(np.round(loss,3), [len(acc),1]), 
                    np.reshape(np.round(val_loss,3), [len(acc),1])])])
    
    pred_data=np.vstack([['TRUE_LABEL','PREDICT_LABEL'], 
                         np.hstack([np.reshape(true_label,[len(predict_label),1]),
                         np.reshape(predict_label,[len(predict_label),1])])])
    fpr_tpr=np.vstack([fpr, tpr])
    
    if stage=='val':
        np.savetxt('VAL_data_'+model_name+'.csv', model_data, delimiter=';', fmt='%s')    
        np.savetxt('VAL_pred_'+model_name+'.csv',pred_data, delimiter=';',fmt='%s')
        np.savetxt('VAL_fpr_tpr_'+model_name+'.csv',fpr_tpr, delimiter=';')
        np.savetxt('times_'+model_name+'.csv',times, delimiter=';')

    else:
        np.savetxt('data_'+model_name+'.csv', model_data, delimiter=';', fmt='%s')    
        np.savetxt('pred_'+model_name+'.csv',pred_data, delimiter=';',fmt='%s')
        np.savetxt('fpr_tpr_'+model_name+'.csv',fpr_tpr, delimiter=';')


