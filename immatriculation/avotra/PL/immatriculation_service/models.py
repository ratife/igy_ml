from django.db import models

# Create your models here.
class Hotel(models.Model):
    name = models.CharField(max_length = 50)
    emp_image = models.ImageField(upload_to='images/')

