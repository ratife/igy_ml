import cv2
import matplotlib.pyplot as plt
import mtcnn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from os import listdir
from os.path import isdir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
#import keras
import pickle
from sklearn.svm import SVC
import utils_fr
detectors = mtcnn.MTCNN()

def decoupe_im(results,im):
    """ Store all face in a liste"""
    all_face = []
    box_coor = []
    for i in range(len(results)):
        x,y,w,h = results[i]['box'] # recupre les bbox
        x1,y1 = abs(x),abs(y)
        x2,y2 = x1+w,y1+h
        face = im[y1:y2,x1:x2]
        all_face.append(face)
        box_coor.append((x,y,w,h))
    return all_face,box_coor

def extract_face(image):
    """Function to detect all face in image"""
    #im = cv2.imread(image) #read image
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #convert image to RGB
    results = detectors.detect_faces(im) #detect all face in image
    all_face,box_coor = decoupe_im(results,im) # store all image in liste
    all_face = [cv2.resize(all_face[i],(160,160)) for i in range(len(all_face))] #resize all image in liste 
    all_face = np.array(all_face)
    return all_face,box_coor

def extract_face_path(image):
    """Function to detect all face in image"""
    im = cv2.imread(image) #read image
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) #convert image to RGB
    results = detectors.detect_faces(im) #detect all face in image
    all_face,box_coor = decoupe_im(results,im) # store all image in liste
    all_face = [cv2.resize(all_face[i],(160,160)) for i in range(len(all_face))] #resize all image in liste 
    all_face = np.array(all_face)
    return all_face,box_coor

def chargement_faces(path):
    faces = []
    #print(listdir(path))
    for x in listdir(path):
        #print("*** " , x)
        path1 = path + x
        face,_ = extract_face_path(path1)
        faces.append(face)
    return faces

def load_dataset(directory):
    """Directory hatrany am train na val"""
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        print(path)
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = chargement_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

def piplene_face(model, face_pixels):
    # scale pixel values
    #face_pixels = face_pixels.reshape(160,160,3)
    
    face_pixels = face_pixels.astype('float32')
    
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    print(samples.shape)
    yhat = model.predict(samples)
    
    return yhat[0]
  
def piplene_face_train(model, face_pixels):
    # scale pixel values
    #face_pixels = face_pixels.reshape(160,160,3)
    
    face_pixels = face_pixels.astype('float32')
    
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    print(samples[0].shape)
    yhat = model.predict(samples[0])

def piplene_face_test(model, face_pixels):
    # scale pixel values
    #face_pixels = face_pixels.reshape(160,160,3)
    
    face_pixels = face_pixels.astype('float32')
    
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    print(samples.shape)
    yhat = model.predict(samples)
    
    return yhat[0]

def extract_face1(image):
    all_face = []
    box_coor = []
    #im = cv2.imread(image) #read image
    im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #convert image to RGB
    faces = face_cascade.detectMultiScale(im, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = im[y:y+h,x:x+w]
        face = cv2.resize(face,(160,160))
        all_face.append(face)
        box_coor.append((x,y,w,h))
    return np.array(all_face),box_coor

def count_liste(liste,text_match):
    l , con = np.unique(liste,return_counts = True)
    return con[list(l).index(text_match)]%2
