#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import time
from tensorflow.keras.models import load_model
from core.utilitaire import *

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

model1 = load_model('core/model/premier_model_unet_epochs_120_scale_128_nb_150.h5', custom_objects={'LossDice': LossDice} )

@csrf_exempt
def WebController(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        base_directory = os.getcwd()
        name_im =filename
        try:
            debut = time.time()
            image,c,numero = demo_sary(os.path.join(base_directory,"media",name_im),128,model1)  

        except:
            return render(request, '../templates/erreur.html')
        image_ecrit = ecriture(image,c,numero)
        final = cv2.imwrite(os.path.join(base_directory,"core","static","test.jpg"),image_ecrit)
        fin = time.time() 

        temps = fin - debut
        print("temps de traitement: %.2f secondes" %temps)
        return render(request, '../templates/sortie.html', {'uploaded_file_url': uploaded_file_url,'numero' : numero,'coordonne' : c,'nom_voiture':filename, 'final':final})
    return render(request, '../templates/simple_upload.html')




