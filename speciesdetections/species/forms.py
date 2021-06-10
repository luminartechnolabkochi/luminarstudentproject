from django import forms


#form
class ImageUploadForm(forms.Form):
    image=forms.FilePathField(path="/home/luminar/Downloads/")
