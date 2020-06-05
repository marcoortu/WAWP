from django.views.generic import CreateView, TemplateView

from .models import TextPattern


class IndexView(TemplateView):
    template_name = 'index.html'


class TextPatternCreateView(CreateView):
    model = TextPattern
    fields = ('text', 'sentiment')
    template_name = 'textpattern/create.html'
