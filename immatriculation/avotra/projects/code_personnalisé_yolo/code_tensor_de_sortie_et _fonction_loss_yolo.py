#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf

La fonction output_yolo:
    Cette fonction retourne la sortie d'un reseau yolo
        output_tensor[i][0] designe la probabilite de l'existance de l'image dans cette bounding box
        output_tensor[i][1],output_tensor[i][2] désigne les coordonnes du milieu du boundingBox
        output_tensor[i][3],output_tensor[i][4] désigne la longueur et la hauteur du boundinbox
        output_tensor[i][5] les classes possibles
# In[3]:


#nbr_bbox = nombre de boundingbox
#nbr_cls = nombre de classe
def output_yolo(nbr_bbox,nbr_cls):
    output_tensor = []
    for i in range (nbr_bbox):
        block = np.zeros(5+nbr_cls)
        output_tensor.append(block)
    return output_tensor
#nbr_grid = nombre de grille
def set_output_yolo(nbr_grid,nbr_bbox,nbr_cls):
    set_output_tensor = []
    for i in range(nbr_grid):
        set_output_tensor.append(output_yolo(nbr_bbox,nbr_cls))
    return set_output_tensor
def square_error_euclidian(a,a_c,b,b_c):
    m = pow((a-a_c),2)
    n = pow((b-b_c),2)
    return m+n


# In[4]:


def first_block_loss(lamba_coord, nbr_grid,nbr_bbox,nbr_cls,x_estimation,y_estimation):
    """  
        - lambda_coord est un entier
        - nbr_grid est le nombre de grille, S^2 dans yolo paper
        - nbr_bbox est le nombre de bounding box, B dans yolo paper
        - nbr_cls est le nombre de classe
        - x_estimation est le x_chapeau dans yolo paper
        - y_estimation est le y_chapeau dans yolo paper
    """
    l1 = []
    s_o_y = set_output_yolo(nbr_grid,nbr_bbox,nbr_cls)
    for i in range(nbr_grid):
        for j in range(nbr_bbox):
            if s_o_y[i][j][5]!=0:
                l1.append(square_error_euclidian(s_o_y[i][j][1],x_estimation,s_o_y[i][j][2],y_estimation))
            else : 
                l1.append(0)
    return lamba_coord*sum (l1)      


# In[5]:


def second_block_loss(lamba_coord, nbr_grid,nbr_bbox,nbr_cls,w_estimation,h_estimation):
    """  
        - lambda_coord est un entier
        - nbr_grid est le nombre de grille, S^2 dans yolo paper
        - nbr_bbox est le nombre de bounding box, B dans yolo paper
        - nbr_cls est le nombre de classe
        - w_estimation est le w_chapeau dans yolo paper
        - h_estimation est le h_chapeau dans yolo paper
    """
    l1 = []
    s_o_y = set_output_yolo(nbr_grid,nbr_bbox,nbr_cls)
    for i in range(nbr_grid):
        for j in range(nbr_bbox):
            if s_o_y[i][j][5]!=0:
                l1.append(square_error_euclidian(sqrt(s_o_y[i][j][3]),sqrt(w_estimation),sqrt(s_o_y[i][j][4]),sqrt(h_estimation)))
            else: 
                l1.append(0)
    return lamba_coord*sum (l1)   


# In[6]:


def third_block_loss(lamba_coord, nbr_grid,nbr_bbox,nbr_cls,c_estimation):
    """  
        - lambda_coord est un entier
        - nbr_grid est le nombre de grille, S^2 dans yolo paper
        - nbr_bbox est le nombre de bounding box, B dans yolo paper
        - nbr_cls est le nombre de classe
        - c_estimation est le c_chapeau dans yolo paper
    """
    l1 = []
    s_o_y = set_output_yolo(nbr_grid,nbr_bbox,nbr_cls)
    for i in range(nbr_grid):
        for j in range(nbr_bbox):
            if s_o_y[i][j][5]!=0:
                l1.append(square_error_euclidian(s_o_y[i][j][5],c_estimation,0,0))
            else: 
                l1.append(0)
    return lamba_coord*sum (l1)   


# In[7]:


def fourth_block_loss(lamba_coord_nonobj, nbr_grid,nbr_bbox,nbr_cls,c_estimation):
    """  
        - lambda_coord est un entier
        - nbr_grid est le nombre de grille, S^2 dans yolo paper
        - nbr_bbox est le nombre de bounding box, B dans yolo paper
        - nbr_cls est le nombre de classe
        - c_estimation est le c_chapeau dans yolo paper
    """
    l1 = []
    s_o_y = set_output_yolo(nbr_grid,nbr_bbox,nbr_cls)
    for i in range(nbr_grid):
        for j in range(nbr_bbox):
            if s_o_y[i][j][5]==0:
                l1.append(square_error_euclidian(s_o_y[i][j][5],c_estimation,0,0))
            else: 
                l1.append(0)
    return lamba_coord_nonobj*sum (l1)   


# In[8]:


def fift_block_loss(nbr_grid,nbr_bbox,nbr_cls):
    """  
        - lambda_coord est un entier
        - nbr_grid est le nombre de grille, S^2 dans yolo paper
        - nbr_bbox est le nombre de bounding box, B dans yolo paper
        - nbr_cls est le nombre de classe
        - c_estimation est le c_chapeau dans yolo paper
    """
    l1 = []
    s_o_y = set_output_yolo(nbr_grid,nbr_bbox,nbr_cls)
    for i in range(nbr_grid):
        if s_o_y[i][j][0]!=0:
            l1.append(square_error_euclidian(s_o_y[i][j][5],c_estimation,0,0))
        else: 
                l1.append(0)
    return lamba_coord_nonobj*sum (l1)   


# In[9]:


def fift():
    return 0


# In[10]:


def loss_fonction_yolo(lamba_coord,lamba_coord_nonobj,nbr_grid,nbr_bbox,nbr_cls,pc_estimation,x_estimation,y_estimation,w_estimation,h_estimation,c_estimation):
    return first_block_loss(lamba_coord, nbr_grid,nbr_bbox,nbr_cls,x_estimation,y_estimation) +            second_block_loss(lamba_coord, nbr_grid,nbr_bbox,nbr_cls,w_estimation,h_estimation) +            third_block_loss(lamba_coord, nbr_grid,nbr_bbox,nbr_cls,c_estimation) +            fourth_block_loss(lamba_coord_nonobj, nbr_grid,nbr_bbox,nbr_cls,c_estimation) + fift()  


# In[11]:


def transform_coordone(x,y,w,h):
    a = x - (w/2)
    b = y + (h/2)
    c = x + (w/2)
    d = y - (h/2)
    return a,b,c,d

def iou(box1,box2):
    """ coordonne de box1
            x1 = box1[0]
            y1 = box1[1]
            x2 = box1[2]
            y2 = box1[3]
            
        coordonne de box1
            x1p = box2[0]
            y1p = box2[1]
            x2p = box2[2]
            y2p = box2[3]        
            
            """
    a = max (box1[0],box2[0])
    b = min (box1[2],box2[2])
    
    c = max (box1[3],box2[3])
    d = min (box1[1],box2[1])
    
    aire_intersection = abs ( a - b ) * abs( c - d )
    aire_union = (abs(box1[0] - box1[2]) * abs(box1[1] - box1[3])) + (abs(box2[0] - box2[2]) * abs(box2[1] - box2[3])) 
    
    response_final = aire_intersection / aire_union
    return response_final


# In[12]:


output_yolo(2,1)


# In[13]:


min(3,1)


# In[14]:


abs(-1)


# In[15]:


def non_max_suppression(*box,threshold):
    """
        Non max suppression:
            - recupere le proba maximum des box
            - iou entre le box contenant le proba max et les autres box
            - supprime les iou superieur à threshold
            - retourne le box avec le plus grand proba et la liste des box < threshold
            - a l'exterieur de cette fonction, si len(liste_box_copy) > 1, \
            faire tourner non_max_suppression(liste_box_copy,threshold), 
            ie non_max_suppression (liste_box_copy[0],liste_box_copy[1],...,liste_box_copy[n], threshold)
            - Si on a qu'une seule classe à predir, il suffit de prendre le box contenant le proba elevé
        
    """
    liste_box=[]
    probability_confidence=[]
    for x in box:
        liste_box.append(x)
    for y in range(len(liste_box)):
        pc.append(liste_box[y][0])
    index_max = np.argmax(probability_confidence)
    true_box = liste_box[index_max]
    liste_box.remove(true_box)
    liste_box_copy = liste_box
    for t in liste_box:
        if iou(true_box,t)>threshold:
            liste_box_copy.remove(t)
    return true_box, liste_box_copy


# In[19]:


from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import add, Activation, BatchNormalization,UpSampling2D,concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2


# In[24]:


def convolution_layer (x,nb_filters, kernels, strides = 1):
    x = Conv2D(nb_filters,kernels,padding = "same",strides = strides,activation='linear') (x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    return x

def residual_block(inputs,nb_filters):
    x = convolution_layer(inputs, nb_filters, (1,1))
    x = convolution_layer(x, 2*nb_filters, (3,3))
    x = add([inputs,x])
    x = Activation('linear')(x)
    
    return x

def repeat_residual_block(inputs, nb_filters, nb_repeat):
    x = residual_block(inputs, nb_filters)
    for i in range(nb_repeat):
        x = residual_block(inputs, nb_filters)
    
    return x

def darknet53(inputs):
    #les deux premiere parties sans residu
    x = convolution_layer(inputs, 32,(3,3))
    x = convolution_layer(x, 64,(3,3),strides = 2)
    #--------------------------------------------------    
    #La premiere block résiduelle
    x = repeat_residual_block(x, 32, nb_repeat=1)
    #--------------------------------------------------
    #--------------------------------------------------    
    #deuxieme couche san residu
    x = convolution_layer(x, 128, (3,3), strides = 2)
    #--------------------------------------------------    
    #deuxieme block residuelle
    x = repeat_residual_block(x, 64, nb_repeat=2)
    #--------------------------------------------------
    #--------------------------------------------------    
    #Troisieme couche sans résidu
    x = convolution_layer(x, 256, (3,3) , strides = 2)
    #--------------------------------------------------
    #Troisieme block residuelle
    x = repeat_residual_block(x, 128, nb_repeat=8)
    #--------------------------------------------------
    #--------------------------------------------------
    #quatrieme couche sans résidu
    x = convolution_layer(x, 512, (3,3), strides=2)
    #--------------------------------------------------
    #quatrieme block residuelle
    x = repeat_residual_block(x, 256, nb_repeat=8)
    #--------------------------------------------------
    #--------------------------------------------------
    #cinquieme couche sans residu
    x = convolution_layer(x, 1024, (3,3), strides=2)
    #--------------------------------------------------
    #cinquieme block residuel
    x = repeat_residual_block(x, 512, nb_repeat= 4)
    
    return x

def darknet_classifier ():
    inputs = Input(shape=(416,416,3))
    x = darknet53(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs,x)
    return model

def yolo_network():
    inputs = Input(shape=(416,416,3))
    #les deux premiere parties sans residu
    x = convolution_layer(inputs, 32,(3,3))
    x = convolution_layer(x, 64,(3,3),strides = 2)
    #--------------------------------------------------    
    #La premiere block résiduelle
    x = repeat_residual_block(x, 32, nb_repeat=1)
    #--------------------------------------------------
    #--------------------------------------------------    
    #deuxieme couche san residu
    x = convolution_layer(x, 128, (3,3), strides = 2)
    #--------------------------------------------------    
    #deuxieme block residuelle
    x = repeat_residual_block(x, 64, nb_repeat=2)
    #--------------------------------------------------
    #--------------------------------------------------    
    #Troisieme couche sans résidu
    x = convolution_layer(x, 256, (3,3) , strides = 2)
    #--------------------------------------------------
    #Troisieme block residuelle
    x = repeat_residual_block(x, 128, nb_repeat=8)    
    skip_36 = x
    #--------------------------------------------------
    #--------------------------------------------------
    #quatrieme couche sans résidu
    x = convolution_layer(x, 512, (3,3), strides=2)
    #--------------------------------------------------
    #quatrieme block residuelle
    x = repeat_residual_block(x, 256, nb_repeat=8)
    skip_64 = x
    #--------------------------------------------------
    #--------------------------------------------------
    #cinquieme couche sans residu
    x = convolution_layer(x, 1024, (3,3), strides=2)
    #--------------------------------------------------
    #cinquieme block residuel
    x = repeat_residual_block(x, 512, nb_repeat= 4)
    #__________________________________________________
    # N O U V E A U  B L O C K
    x = convolution_layer(x, 512, (1,1), strides=1)
    x = convolution_layer(x, 1024, (3,3), strides=1)
    x = convolution_layer(x, 512, (1,1), strides=1)
    x = convolution_layer(x, 1024, (3,3), strides=1)
    x = convolution_layer(x, 512, (1,1), strides=1)
    #vers yolo1
    y1 = x
    y1 = convolution_layer(y1, 1024, (3,3), strides=1)
    y1 = convolution_layer(y1, 255, (1,1), strides=1)
    #continuite du reseau
    x = convolution_layer(x, 256,(1,1),strides=1)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_64])
    
    x = convolution_layer(x, 256, (1,1), strides=1)
    x = convolution_layer(x, 512, (3,3), strides=1)
    x = convolution_layer(x, 256, (1,1), strides=1)
    x = convolution_layer(x, 512, (3,3), strides=1)
    x = convolution_layer(x, 256, (1,1), strides=1)
    
    #vers yolo
    y2 = x
    y2 = convolution_layer(y2, 512, (3,3), strides=1)
    y2 = convolution_layer(y2, 255, (1,1), strides=1)
    #continuite du reseau
    x = convolution_layer(x, 128, (1,1), strides=1)
    x= UpSampling2D(2)(x)
    x = concatenate([x,skip_36])
    
    y3 = convolution_layer(x, 128, (1,1), strides=1)
    y3 = convolution_layer(y3, 256, (3,3), strides=1)
    y3 = convolution_layer(y3, 128, (1,1), strides=1)
    y3 = convolution_layer(y3, 256, (3,3), strides=1)
    y3 = convolution_layer(y3, 128, (1,1), strides=1)
    y3 = convolution_layer(y3, 256, (3,3), strides=1)
    y3 = convolution_layer(y3, 255, (1,1), strides=1)
    
    model = Model(inputs, [y1,y2,y3])
    
    return model
    


# In[61]:


import os
os.chdir("E:/Image_test/test")


# In[62]:


import cv2


# In[67]:


im = cv2.imread("4.jpg")
imr = cv2.resize(im , (416,416))


# In[68]:


model1 = darknet_classifier()


# In[65]:


model1.fit()


# In[25]:


model1 = yolo_network()


# In[26]:


model1.summary()


# In[27]:


80000/50


# In[ ]:




