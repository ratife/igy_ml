import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pytesseract
import glob
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from fusion import fsplit
from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import time
import pickle

from utilis_pl import *

#import model
os.chdir("E:/model_enregistre")
with open("oc_isol.pkl", 'rb') as file:
    if_clf_model= pickle.load(file)
    
#Se déplace vers l'IMAGE DE TEST
os.chdir("E:/Image_test")
image = cv2.imread("12.jpg")
gh,im,pos = image_detecte_plaque(image) #(gh est egal à image, im est l'ensemble de toutes les image en forme de rectangle dans notre test  et pos sont les coordonnées du bounding rect )
#resize de l'ensemble des rectangle  en 244x244
im = [cv2.resize (im[i] , (224,224)) for i in range(len(im))]
im = np.array(im)
#feature extraction via resnet
im_resnet50 = Resnet50(im)
prediction = if_clf_model.predict(im_resnet50)

###################ocr##############################
s=[]
for c,x in enumerate(prediction):
    if x == 1:
        s.append(c)     
d=[]
for u in range(len(prediction)):
    d.append('')
custom_config = r'--oem 3 --psm 6'
im = [cv2.resize(im[i] , (700,200)) for i in range(len(im))]

#CONVERTIT TOUTES LES PLAQUES PREDITS EN String
for nbr,p in enumerate(s):
    im[p] = cv2.cvtColor(im[p],cv2.COLOR_BGR2GRAY)
    d[nbr] = pytesseract.image_to_string(im[p], config=custom_config)

 #Ecriture de la sortie en string sur l'image de la plaque predite   
for p in s:
    cv2.rectangle(image,(pos[p][0],pos[p][1]),(pos[p][0]+pos[p][2],pos[p][1]+pos[p][3]),(0,0,255),2)    
    cv2.putText(image, d[p], (pos[p][0],pos[p][1]), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 100, 50), 1)  


#cv2.imwrite("test_pl_spyder21.jpg", image)
print(plt.imshow(image))
plt.show()