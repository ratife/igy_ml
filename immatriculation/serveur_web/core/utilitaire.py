import tensorflow as tf
import pytesseract
import numpy as np
import cv2
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def LossDice(y_true, y_pred):
    numerateur  =tf.reduce_sum(y_true*y_pred, axis=(1, 2))
    denominateur=tf.reduce_sum(y_true+y_pred, axis=(1, 2))
    dice=2*numerateur/(denominateur+1E-4)
    return 1-dice

def preprocess_image_unet(name_image,size_im):
    a = cv2.imread(name_image)
    a = cv2.resize(a, (size_im,size_im))
    a1 = a/255
    return a,a1

def prediction(name_image,size_im,model):    
    _,y= preprocess_image_unet(name_image,size_im)
    im_pred=model.predict(y.reshape(1,size_im,size_im,3))
    return im_pred.reshape(size_im,size_im)  

def prediction_v(im,size_im):    
    im = cv2.resize(im, (size_im,size_im))
    im_pred=model.predict(im.reshape(1,size_im,size_im,3))
    return im_pred.reshape(size_im,size_im) 

def convert_binaire(im):
    im[im<0.9] = 0
    im[im>0.9] = 1
    return im

def inverse_color(im,size_im):
    for i in range(size_im):
        for j in range(size_im):
            if im[i][j] == 0:
                im[i][j] = 1
            else:
                im[i][j] = 0 
    return im    

def find_contour(im,size_im):
    contours, hierarchy = cv2.findContours(np.uint8(im), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coord = []
    for j in contours:
        (x,y,w,h)=cv2.boundingRect(j)
        if (w<size_im) :
            coord.append((x,y,w,h))
    return np.array(coord)/size_im

def select_ocr(liste):
    ocr_izy = []
    for tttt in liste:
        ls1 = re.findall("[a-zA-Z0-9]", tttt)
        ls2 = ('').join(ls1)
        ocr_izy.append(ls2)
    return "".join(ocr_izy)

def ecriture(image,c,numero):
    cv2.rectangle(image,(c[0],c[1]),(c[0]+c[2],c[1]+c[3]),(0,0,255),2)    
    return cv2.putText(image, numero, (c[0],c[1]), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 100, 50), 1)

def demo_sary (idd,size_im,model):
    im_pred = prediction(idd,size_im,model)
    im_pred_binaire = convert_binaire(im_pred)

    img_dilate = inverse_color(im_pred_binaire,size_im)
    coordonne = find_contour(img_dilate,size_im)
    image = cv2.imread(idd)

    coordonne = coordonne * [image.shape[1],image.shape[0],image.shape[1],image.shape[0]]
    coordonne = coordonne.astype(int)
    plaque = []
    for x in coordonne:
        plaque.append(image[x[1]:x[1]+x[3],x[0]:x[0]+x[2]])
    custom_config = r'--oem 3 --psm 6'
    num_plaque = pytesseract.image_to_string(plaque[0], config=custom_config)
    return image,coordonne[0],select_ocr(num_plaque) 

def demo_video(im,size_im):
    im_pred = prediction_v(im,size_im)
    im_pred_binaire = convert_binaire(im_pred)

    img_dilate = inverse_color(im_pred_binaire,size_im)
    coordonne = find_contour(img_dilate,size_im)
    image = cv2.imread(str(idd)+".jpg")

    coordonne = coordonne * [image.shape[1],image.shape[0],image.shape[1],image.shape[0]]
    coordonne = coordonne.astype(int)
    plaque = []
    for x in coordonne:
        plaque.append(image[x[1]:x[1]+x[3],x[0]:x[0]+x[2]])
    custom_config = r'--oem 3 --psm 6'
    num_plaque = pytesseract.image_to_string(plaque[0], config=custom_config)
    return coordonne[0],select_ocr(num_plaque) 