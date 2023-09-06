#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import json
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from core.utilitaire import *

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

model1 = load_model('core/model/premier_model_unet_epochs_120_scale_128_nb_150.h5', custom_objects={'LossDice': LossDice} )

@csrf_exempt
def WSController(request):
	if request.method == 'POST' and request.FILES['myfile']:
		myfile = request.FILES['myfile']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)
		base_directory = os.getcwd()
		name_im =filename
		if name_im:
		#transcript = django_tesseract.tesseract.transcript(file_obj)
			coo = []
			image,c,numero = demo_sary(os.path.join(base_directory,"media",name_im),128,model1)
			return JsonResponse({'numero':numero}, status=200)
		return JsonResponse({}, status=204)

		