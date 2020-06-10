from django import forms

from .models import TextPattern


class TextPatternUpdateForm(forms.ModelForm):
    id = forms.IntegerField(widget=forms.HiddenInput())

    class Meta:
        model = TextPattern
        fields = ('text', 'sentiment')
        widgets = {
            'text': forms.Textarea(attrs={'readonly': 'readonly'}),
        }


class TextPatternClassifyForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, label='Testo')

    class Meta:
        widgets = {
            'text': forms.Textarea(),
        }


class TextPatternForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, label='Testo')
