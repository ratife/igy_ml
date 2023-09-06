import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract
import time
os.chdir("E:/Image_test/test_2")

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

def preprocess_ocr(liste):
    ocr = [cv2.cvtColor(liste[i], cv2.COLOR_BGR2GRAY) for i in range(len(liste))]
    ocr_resize = [cv2.resize(ocr[i], (700,200)) for i in range(len(liste)) ]
    return np.array(ocr_resize)


def select_ocr1(liste):
    ocr_izy = []
    for tttt in liste:
        ls1 = re.findall("[a-zA-Z0-9]", tttt)
        ls2 = ('').join(ls1)
        ocr_izy.append(ls2)
    return ocr_izy


start = time.clock()
image = cv2.imread("12.jpg") # Charge l'image
im_origine, many_rectangle, position_rectangle = image_detecte_plaque(image) # Detecte les rectangles et les positions 
many_rectangle_array = np.array(many_rectangle) # Variable qui stock les rectangles (conversion en array)
position_rectangle_array = np.array(position_rectangle) # Variable qui stock les positions des rectangles(conversion en array)
resize_ocr = preprocess_ocr(many_rectangle_array) # Grayscale and resize all rectangle image in 700x200 
custom_config = r'--oem 3 --psm 6' # Configuration ocr
ocr = [pytesseract.image_to_string(resize_ocr[p], config=custom_config) for p in range(len(resize_ocr))] # Convertis l'image en texte
ocr_selected = select_ocr1(ocr) # Selectionne que les nombre et alphabet de la sortie ocr
########################################################
#NE TRIE QUE LES TEXTES PERTINANTES
index_selected = [] 
for cnt,d in enumerate(ocr_selected):
    if len (d) > 5 and len (d) <10:
        index_selected.append(cnt)    
#FIN TRIAGE TEXTE PERTINANTE ET RETOURNE CES INDEX
######################################################
#TRIE LES PLAQUES PERTINANTES
ff = []
true_index = []
i=0
for j1 in index_selected:
    for j in range(10):
        for e1 in ocr_selected[j1]:
            if e1 == str(j):
                i = i+1
    if i>2:
        true_index.append(j1)
    i=0
#FIN TRIAGE PLAQUE PERTINANTE
#######################################################
#plot les plaques
for m1 in true_index:
    cv2.rectangle(image,(position_rectangle_array[m1][0],position_rectangle_array[m1][1]),(position_rectangle_array[m1][0]+position_rectangle_array[m1][2],position_rectangle_array[m1][1]+position_rectangle_array[m1][3]),(0,0,255),2)    
    cv2.putText(image, ocr_selected[m1], (position_rectangle_array[m1][0],position_rectangle_array[m1][1]), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 100, 50), 1)  
plt.imshow(image)
end= time.clock()
print("Le temp d'excecution est de ",end-start,"seconde")