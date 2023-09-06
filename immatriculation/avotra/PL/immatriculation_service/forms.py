#forms.py
from django import forms
from .models import *

class HotelForm(forms.ModelForm):
    class meta:
        model = Hotel
        fields = ['name', 'hotel_Main_Img']


