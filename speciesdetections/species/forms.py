from django import forms



class ImageUploadForm(forms.Form):
    image=forms.FilePathField(path="/home/luminar/Downloads/")