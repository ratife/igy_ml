import cv2
import matplotlib.pyplot as plt
import mtcnn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from os import listdir
from os.path import isdir
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
import keras
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

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

def getPreparatedData():
    directory_train = "E:/face_recognition/dataset_complet_train_colab_test1603/"
    X_train,y_train = load_dataset(directory_train)

    X_train1 = []
    vide = []

    for i in range(len(X_train)):
        if X_train[i].shape[0]!=0:
            a = X_train[i][0]
            X_train1.append(a)
        else:
            vide.append(i)

    y_train_izy = y_train

    ii = 0
    for c in vide:
        print(c)
        y_train_izy = np.delete(y_train_izy,c-ii)
        ii = ii+1

    Y_train = y_train_izy
    X_Train = X_train1

    X_standardiser = []
    for x in X_Train:
        m,st = x.mean(),x.std()
        x = (x-m)/st
        X_standardiser.append(x)  

    return Y_train,X_standardiser


def train():
    nom_model_facnet = "E:/face_recognition/facenet_keras.h5"
    model_facenet = load_model(nom_model_facnet)
    Y_train,X_standardiser = getPreparatedData()

    X_feature = model_facenet.predict(np.array(X_standardiser))

    input_encoder = Normalizer(norm='l2')

    out_encoder = LabelEncoder()

    X_feature_n = input_encoder.fit_transform(X_feature)

    Y_train_n = out_encoder.fit_transform(Y_train)

    Y_train_n = out_encoder.fit_transform(Y_train)

    #encoder='encoder_personnel_igy_2021_modif_16_03.pkl'
    #with open(encoder, 'wb') as file1:
    #    pickle.dump(out_encoder, file1)

    model_svm= SVC(kernel='linear', probability=True)
    model_svm.fit(X_feature_n, Y_train_n)

    model_svm_rbf= SVC(kernel='rbf', probability=True)
    model_svm_rbf.fit(X_feature_n, Y_train_n)

    model_svm_poly= SVC(kernel='poly', degree = 3,probability=True)
    model_svm_poly.fit(X_feature_n, Y_train_n)

    
    yhat_train = model_svm.predict(X_feature_n)
    score_train = accuracy_score(Y_train_n, yhat_train)

    from sklearn.metrics import accuracy_score
    yhat_train = model_svm_rbf.predict(X_feature_n)
    score_train = accuracy_score(Y_train_n, yhat_train)

    from sklearn.metrics import accuracy_score
    yhat_train = model_svm_poly.predict(X_feature_n)
    score_train = accuracy_score(Y_train_n, yhat_train)

    #RNA Train
    import tensorflow as tf
    model_igy = tf.keras.models.Sequential()
    model_igy.add(tf.keras.Input(shape=(128,)))
    model_igy.add(tf.keras.layers.Dense(86, activation='sigmoid'))
    import tensorflow.keras
    model_igy.compile(
        optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'],
    )
    model_igy.fit(X_feature_n,Y_train_n,epochs=35,batch_size=1)


    with open("svm_pers_igy_2021_sans_normalizer_modiff_1.pkl", 'wb') as file1:
    	pickle.dump(model_svm_sans_normalizer, file1)
 
    with open("svm_pers_igy_2021_16_03.pkl", 'wb') as file1:
    	pickle.dump(model_svm, file1)

    model_igy.save("rna_igy_modiff.h5")



def use():
    info = []
    ar=[]
    image = cv2.imread("E:/face_recognition/test_face_recognition_igy/15085581_643487875824188_6701308157781263355_n.jpg")
    all_faces,box_coor = extract_face(image)
    for i,face in enumerate(all_faces):
        moyenne,variance = face.mean(),face.std()
        face = (face - moyenne) / variance
        face = face.reshape(1,160,160,3)
        prediction = model_facenet.predict(face)
        #prediction = input_encoder.fit_transform(prediction)
        prediciton_face = model_svm_poly.predict(prediction)
        prediciton_face_proba = model_svm_poly.predict_proba(prediction)
        aa,bb = np.argmax(prediciton_face_proba),np.max(prediciton_face_proba)
        print(bb)
        if bb > 0.9999:
	        #raha mipredir anarana
	        x,y,w,h = box_coor[i]
	        print(prediciton_face)
	        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
	        print( out_encoder.inverse_transform(np.array(prediciton_face))[0])
	        cv2.putText(image, out_encoder.inverse_transform(np.array(prediciton_face))[0], (x,y), cv2.FONT_HERSHEY_DUPLEX,1, (255, 255, 0), 1) 

        
    plt.figure(figsize=(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)


#getPreparatedData()