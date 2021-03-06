from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.views import View
from django.views.generic import CreateView, TemplateView, UpdateView

from .sentiment import ItalianSentimentAnalyzer
from .forms import TextPatternUpdateForm, TextPatternClassifyForm
from .models import TextPattern


class IndexView(TemplateView):
    template_name = 'index.html'


class TextPatternCreateView(CreateView):
    model = TextPattern
    fields = ('text',)
    template_name = 'textpattern/create.html'


class LabelTextView(UpdateView):
    model = TextPattern
    form_class = TextPatternUpdateForm
    initial = {'key': 'value'}
    template_name = 'textpattern/label.html'

    def get(self, request, *args, **kwargs):
        text = TextPattern.objects.filter(sentiment=None).first()
        initials = {'id': text.id, 'text': text.text}
        form = self.form_class(initial=initials)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        instance = TextPattern.objects.get(pk=request.POST['id'])
        form = self.form_class(request.POST, instance=instance)
        if form.is_valid():
            form.save()
            ItalianSentimentAnalyzer.train()
            return HttpResponseRedirect('/label/')
        return render(request, self.template_name, {'form': form})


class ClassifyTextView(View):
    template_name = 'classify.html'

    def get(self, request, *args, **kwargs):
        form = TextPatternClassifyForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = TextPatternClassifyForm(request.POST)
        if form.is_valid():
            clf = ItalianSentimentAnalyzer()
            predicted = clf.predict(form.cleaned_data['text'])
            return render(
                request, self.template_name,
                {
                    'form': form,
                    'predicted': predicted['sentiment'],
                    'confidence': predicted['probability']
                }
            )
        return render(request, self.template_name, {'form': form})
