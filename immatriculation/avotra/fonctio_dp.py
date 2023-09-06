import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("E:/videos_test")

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
    n,gray_binaire=cv2.threshold(gray,0,255,cv2.THRESH_OTSU +  cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(gray_binaire,50,150,apertureSize = 5)
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
    liste = []
    kernel = np.ones((5, 5), np.uint8)
    contour=prepocessing_image(image)
    for i,j in enumerate(contour):
        (x,y,w,h)=cv2.boundingRect(j)
        rapport = w/h
        produit = w*h
        if (rapport >=2 and rapport<=5 and produit>300) or (rapport >=1 and rapport<=2 and produit >300) :
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            #cv2.putText(image, "", (x,y), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 100, 50), 1)  
            rr_i = image[y:y+h,x:x+w]
            img_resize = cv2.resize(rr_i, (0, 0), fx=10, fy=10)
            d= (x,y,w,h)
            liste.append((rr_i,d))
            print("")
            #img_corrige = cv2.morphologyEx(img_resize, cv2.MORPH_OPEN, kernel)
            #img_corrige = cv2.GaussianBlur(img_corrige,(3, 3), 3)
            #cv2.imwrite("image-detecte-protocole-2"+str(i)+".png",img_corrige)   
    return liste
