# -*- coding: utf-8 -*-
"""

@author: Andres Agreda

"""

import numpy as np
from funciones import llamado, normalizacion, detectorQRS, segmentacion, filtro, Binarizacion, completar, ajusteDatos

#se definen Variables
etq=[]
dts=[]
etiquetas=[]
etiquetas1=[]
etiquetas2=[]
datosSeg=[]
datosSegmentados=[]

cta_registros=1

#Se recorren los primeros 20 registros de la base de datos
for re in range(cta_registros):
    
    datosSeg=[]
    etiquetas1=[]
       
    #se llama a los registros
    raw_data, id_events=llamado(re)
    
    #se normaliza
    data_normal=normalizacion(raw_data)
    
    #se hace uso del detector QRS
    qrs_locs=detectorQRS(re)
    
    #se hace uso de los indices arrojados por el detector QRS
    p=segmentacion(data_normal, qrs_locs)
       
    datosSeg.append(p)
    
    for i in range(len(datosSeg)):
        for j in range(len(datosSeg[i])):
            datosSegmentados.append(datosSeg[i][j])
    
    ll=filtro(datosSegmentados)
    #Binarizacion de las etiquetas
    etiquetas1.append(Binarizacion(id_events))
    
    
    for i in range(len(etiquetas1)):
        for j in range(len(etiquetas1[i])):
            etiquetas2.append(etiquetas1[i][j])
    """
    se completan los datos y las etiquetas con el fin de que tengan la misma 
    la misma longitud y se realiza el filtro de los mismos
    """
    etq, dts = completar(etiquetas2, datosSegmentados)    

dtsCom=[]
etqCom=[]

#se ubica en una sola lista los datos de todos los regitros, asi mismo con las etiquetas
for i in range(len(etq)):
    for j in range(len(etq[i])):
        if(etq[i][j]==1):
        
            dtsCom.append(dts[i][j])
            etqCom.append(1)
        else:
            
            dtsCom.append(dts[i][j])
            etqCom.append(0)

#se ajusta los segmentos de datos para que tengan la misma longitud            
dtsCom=ajusteDatos(dtsCom)

#Se guarda los datos
np.savetxt('datosFinal/datos.dat',dtsCom)
np.savetxt('datosFinal/etiquetas.dat',etqCom)


