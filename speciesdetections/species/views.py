from django.shortcuts import render
from .forms import ImageUploadForm
# Create your views here.
import requests
import sys
from subprocess import run,PIPE

# os.system('path')
def predict_species(request):
    forms=ImageUploadForm()
    context={}
    context["form"]=forms
    if request.method == "POST":
        form=ImageUploadForm(request.POST,files=request.FILES)
        print("inside post")
        if form.is_valid():
            print("here")
            inpt=form.cleaned_data.get("image")
            out=run([sys.executable,'/home/luminar/Downloads/singletest/single_test.py',inpt],shell=False,stdout=PIPE)
            data=str(out.stdout)
            res=data.lstrip("b")
            print(res)
            context["data"]=res
            return render(request, "index.html", context)




    return render(request,"index.html",context)