from django.db import models
from django.utils.timezone import now

SENTIMENT = [
    ('positive', 'Positive'),
    ('neutral', 'Neutral'),
    ('negative', 'Negative'),
]


class TextPattern(models.Model):
    date = models.DateTimeField(
        default=now
    )
    sentiment = models.CharField(
        max_length=30,
        blank=True,
        null=True,
        choices=SENTIMENT,
        default=SENTIMENT[1][0]
    )
    text = models.TextField(
        blank=True
    )
