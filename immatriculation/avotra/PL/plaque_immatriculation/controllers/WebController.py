from django.shortcuts import render


from django.http import HttpResponseRedirect,HttpResponse
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from immatriculation_service.forms import HotelForm

from django.views.generic import DetailView
from immatriculation_service.models import Hotel

# Create your views here.
class WebController(TemplateView):
    def hotel_image_view(request):  
        if request.method == 'POST': 
            form = HotelForm(request.POST, request.FILES) 
      
            if form.is_valid(): 
                form.save() 
                return redirect('success') 
        else: 
            form = HotelForm() 
        return render(request, 'hotel_image_form.html', {'form' : form}) 
      
  
    def success(request): 
        return HttpResponse('successfully uploaded') 
        form = HotelForm
        template_name = 'upload_image.html'

    def post(self, request, *args, **kwargs):

        form = HotelForm(request.POST, request.FILES)

#ici intervenir le request

        if form.is_valid():
            obj = form.save()

            return HttpResponseRedirect(reverse_lazy('display_image', kwargs={'pk': obj.id}))

        context = self.get_context_data(form=form)
        return self.render_to_response(context)     

    def get(self, request, *args, **kwargs):
        #print("fefzfe")
        return render(request, '../templates/display_image.html')

#class ImageDisplay(DetailView):
 #  model = Image_service
  #  template_name = 'display_image.html'
   # context_object_name = 'emp'



