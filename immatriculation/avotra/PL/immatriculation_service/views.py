#from django.shortcuts import render


#from django.http import HttpResponseRedirect,HttpResponse
#from django.urls import reverse_lazy
#from django.views.generic import TemplateView
#from immatriculation_service.forms import ImageForm

#from django.views.generic import DetailView
#from immatriculation_service.models import Image_service

# Create your views here.

#class ImmatriculationImage(TemplateView):

 #   form = ImageForm
  #  template_name = 'upload_image.html'

   # def post(self, request, *args, **kwargs):

    #    form = ImageForm(request.POST, request.FILES)

#ici intervenir le request

#        if form.is_valid():
 #           obj = form.save()

  #          return HttpResponseRedirect(reverse_lazy('display_image', kwargs={'pk': obj.id}))

   #     context = self.get_context_data(form=form)
    #    return self.render_to_response(context)     

    #def get(self, request, *args, **kwargs):
        #print("fefzfe")
    #    return HttpResponse("fdsqfq")
        #return self.post(request, *args, **kwargs)

#class ImageDisplay(DetailView):
 #   model = Image_service
  #  template_name = 'display_image.html'
#    context_object_name = 'emp'



