from django.contrib import admin
from .models import TextPattern


@admin.register(TextPattern)
class TextPatternAdmin(admin.ModelAdmin):
    list_display = ('sentiment', 'text')
