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

def prepocessing_image(image):
    """
    Fonction " prepocessing_image":
    Input : Image
    Construction:
        --- Conversion de l'image en niveau de gris avec la fonction cv2.cvtColor
        --- Seuillage de l'image recemment convertie avec le suillage binaire et OTSU
        --- Tracage des contours de l'image qui vient d'être seuillé
                    parametre: 50,150,apertureSize = 5
        --- Detection de contour via cv2.findContours
    Output : Les contours detecté dans l'image
    
    
    """
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.bitwise_not(gray)
    #n,gray_binaire=cv2.threshold(gray,0,255,cv2.THRESH_OTSU +  cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray,0,255,apertureSize = 7)
    contours,h =cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def image_detecte_plaque(image):
    """
    
    Fonction "image_detecte_plaque":
    Input : Image
    Construction:
        --- Appel à la fonction preprocessing_image
        --- Pour chaque contour, on selectionne les coordonnées via cv2.boundingRect
        --- On retient que les candidats ayant un rapport longeur/largeur entre 2 et 5 ou 1 et 2 et un produit>300
                Ceci afin de se débarasser des tres petites contours
    Output: Les contour plus grand enforme de rectangle ou carrée de l IMAGE
                                           
    """ 
    all_im = []
    pos = []
    kernel = np.ones((5, 5), np.uint8)
    contour=prepocessing_image(image)
    for i,j in enumerate(contour):
        (x,y,w,h)=cv2.boundingRect(j)
        rapport = w/h
        produit = w*h
        if (rapport >=1 and rapport<=5 and produit>300) :
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            #cv2.putText(image, "", (x,y), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 100, 50), 1)  
            rr_i = image[y:y+h,x:x+w]
            all_im.append(rr_i)
            pos.append((x,y,w,h))
            #img_resize = cv2.resize(rr_i, (0, 0), fx=10, fy=10)
            #img_corrige = cv2.morphologyEx(img_resize, cv2.MORPH_OPEN, kernel)
            #img_corrige = cv2.GaussianBlur(img_corrige,(3, 3), 3)
            #cv2.imwrite("WQQQ"+str(i)+".png",img_corrige)  
    return image,all_im,pos

def extract_resnet(X,a):  
    # X : images numpy array
    resnet_model = ResNet50(input_shape=a, include_top=False)  # Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(X)
    return features_array

def enter(data_name):
    #resnet_model = ResNet50(input_shape=(224, 224, 3), include_top=False)  # Since top layer is the fc layer used for predictions.shape
    A = [cv2.imread(data_name[i]) for i in range(len(data_name))]
    A1 = [cv2.resize(A[i], (224,224)) for i in range(len(data_name))]
    A1 = np.array(A1)
    #A2 = resnet_model.predict(A1)
    #A3 = A2.reshape(A2.shape[0],A2.shape[1]*A2.shape[2]*2048)
    return A1

def Resnet50(data):
    resnet_model = ResNet50(input_shape=(224, 224, 3), weights="imagenet", include_top=False)  # Since top layer is the fc layer used for predictions.shape
    A2 = resnet_model.predict(data)
    A3 = A2.reshape(A2.shape[0],A2.shape[1]*A2.shape[2]*2048)
    return A3

def entree1(data_name):
    resnet_model = ResNet50(input_shape=(224, 224, 3), include_top=False)  # Since top layer is the fc layer used for predictions.shape
    #A = [cv2.imread(data_name[i]) for i in range(len(data_name))]
    A1 = [cv2.resize(data_name[i], (224,224)) for i in range(len(data_name))]
    A1 = np.array(A1)
    A2 = resnet_model.predict(A1)
    A3 = A2.reshape(A2.shape[0],A2.shape[1]*A2.shape[2]*2048)
    return A3
    
    
def prediction_pl(chemin, image,estimateur):
    os.chdir(chemin)
    image = cv2.imread("9.jpg")
    gh,im,pos = image_detecte_plaque(image)
    im = [cv2.resize (im[i] , (224,224)) for i in range(len(im))]
    im = np.array(im)
    im_pred = Resnet50(im)
    prediction = estimateur.predict(im_pred)
    return im,pos, prediction

def ocr (image,im,pos):
    d=[]
    for u in range(len(prediction)):
        d.append('')
    custom_config = r'--oem 3 --psm 6'
    im = [cv2.resize(im[i] , (700,200)) for i in range(len(im))]
    for manisa,p in enumerate(s):
        d[manisa] = pytesseract.image_to_string(im[p], config=custom_config)   
    for p in s:
        cv2.rectangle(image,(pos[p][0],pos[p][1]),(pos[p][0]+pos[p][2],pos[p][1]+pos[p][3]),(0,0,255),2)    
        cv2.putText(image, d[p], (pos[p][0],pos[p][1]), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 100, 50), 1)      
    return image