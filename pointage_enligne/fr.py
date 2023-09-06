import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.svm import SVC
from utils_fr import *
import mysql.connector
import time
from datetime import datetime

nom_model_svm = "F:/face_farany/model_0503/svm_pers_igy_2021_0503.pkl"
nom_model_facenet = "F:/face_farany/facenet_keras.h5"
nom_encode = "F:/face_farany/model_0503/encoder_personnel_igy_2021_0503.pkl"

model_facenet = load_model(nom_model_facenet)
with open(nom_model_svm, 'rb') as file:
    model_svm = pickle.load(file)
with open(nom_encode, 'rb') as file:
    out_encoder = pickle.load(file)

def diff_heure(b,a):
    w = [float(str(b-a).split(':')[i]) for i in range(3)]
    return(w[0]*60)+(w[1]*60)+w[2]
def insert_info_login(id_emp,date_crea):
    mydb = mysql.connector.connect(host="localhost",user="root",password="",database="avotrah")
    mycursor = mydb.cursor()
    mycursor.execute('insert into mouvement (employee_id,login,date_mouvement) values (%s,%s,%s)',(id_emp,1,date_crea))
    mydb.commit()

def insert_info_logout(id_emp,date_crea):
    mydb = mysql.connector.connect(host="localhost",user="root",password="",database="avotrah")
    mycursor = mydb.cursor()
    mycursor.execute('insert into mouvement (employee_id,logout,date_mouvement) values (%s,%s,%s)',(id_emp,1,date_crea))
    mydb.commit()


ar=[]
liste_check_in=[]
liste_name = []
cap = cv2.VideoCapture(0)
while True:
    _,image = cap.read()
    image = cv2.resize(image,(0,0),fx=0.5,fy=0.5)
    all_faces,box_coor = extract_face(image)
    gg=0
    if len(all_faces)>0:   
        for i,face in enumerate(all_faces):
            moyenne,variance = face.mean(),face.std()
            face = (face - moyenne) / variance
            face = face.reshape(1,160,160,3)
            prediction = model_facenet.predict(face)
            prediciton_face = model_svm.predict(prediction)
            prediciton_face_proba = model_svm.predict_proba(prediction)
            aa,bb = np.argmax(prediciton_face_proba),np.max(prediciton_face_proba)
            if bb > 0.9999:
                gg=gg+1
                x,y,w,h = box_coor[i]
                b = datetime.fromtimestamp(time.time())
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
                cv2.putText(image, out_encoder.inverse_transform(np.array(prediciton_face))[0], (x,y), cv2.FONT_HERSHEY_DUPLEX,1, (255, 255, 0), 1)
                #cv2.putText(image, 'bienvenue: ' + out_encoder.inverse_transform(np.array(prediciton_face))[0], (int(image.shape[0]/35),int(image.shape[1]/7-gg)), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 0), 1)     
                cv2.putText(image, 'bienvenue: ', (10,10), cv2.FONT_HERSHEY_DUPLEX,1, (0, 0, 255), 1)
                cv2.putText(image,  out_encoder.inverse_transform(np.array(prediciton_face))[0], (int(image.shape[0]/(10)),int(image.shape[1]/(7-gg))), cv2.FONT_HERSHEY_DUPLEX,1, (0, 0, 255), 1)

                i=0
                emp = (prediciton_face[0], b)
                print(emp,str(emp[1]))
                if emp[0] in liste_name:
                    b1 = datetime.fromtimestamp(time.time())
                    while i < len(liste_check_in):
                        if emp[0] == liste_check_in[i][0]:
                            if diff_heure(b1,liste_check_in[i][1]) > 1:
                                insert_info_logout(int(emp[0]),str(b1))
                                del (liste_name[i])
                                break
                        else:
                            i = i+1
                else:
                    liste_name.append(prediciton_face[0])
                    liste_check_in.append((prediciton_face[0],b))
                    insert_info_login(int(emp[0]),str(emp[1]))  
            cv2.imshow('PRESENCE_IGY',image)
    else:
        cv2.imshow('PRESENCE_IGY',image)
    if cv2.waitKey(1) & 0xFF == ord('Q'.lower()):
        break
cap.release()
cv2.destroyAllWindows()

