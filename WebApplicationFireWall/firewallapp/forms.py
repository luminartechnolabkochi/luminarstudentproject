from django import forms


class FirwallCheckForm(forms.Form):
    url_path=forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    #   myfield = forms.CharField(widget=forms.TextInput(attrs={'class': 'myfieldclass'}))