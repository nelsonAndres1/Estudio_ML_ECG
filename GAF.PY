
#Codigo para colab
from google.colab import drive
drive.mount('/content/drive')

import os  
path_base = os.getcwd()
data_base= path_base+'/drive/My Drive/proyecto_ECG/datos_ECG'

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint
import numpy as np
from google.colab.patches import cv2_imshow

X= np.loadtxt(data_base+'/noarritmiafull.dat')
#X= np.loadtxt(data_base+'/arritmiafull.dat')

#SUMMATION
gasf = GramianAngularField(image_size=256, method='summation')

X_gasf = gasf.fit_transform(X)


#DIFFERENCE
X_gasf = []
gasf=[]

gasf = GramianAngularField(image_size=256, method='difference')

X_gasf = gasf.fit_transform(X)


#FIGURES
r = plt.gcf().canvas.get_renderer()
import cv2

#seleccionar metodo segun corresponda

#file_path=data_base+f"/summation/noarritmia/summnoarr_"
#file_path=data_base+f"/summation/arritmia/summarr_"
file_path=data_base+f"/difference/noarritmia/diffnoarr_"
#file_path=data_base+f"/difference/arritmia/diffarr_"

for n,image in enumerate(X_gasf):
    fig = plt.figure(figsize=(8, 4))
    im = plt.imshow(image,cmap='rainbow', origin='lower');
    t=im.make_image(r)
    rgbImage = cv2.cvtColor(t[0], cv2.COLOR_RGBA2RGB)
    #cv2_imshow(rgbImage)
    resized = cv2.resize(rgbImage, (128,128), interpolation = cv2.INTER_AREA)
    cv2.imwrite(file_path+str(n)+'.png',resized)
    plt.clf()
    plt.close("all")
