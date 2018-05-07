from django.db import models
import datetime
from django.forms import model_to_dict

class Product(models.Model):
    title = models.CharField(max_length=1024, default='')
    asin = models.CharField(max_length=1024, default='', primary_key=True)
    last_crawl_date = models.DateField(default=datetime.date.today)
    last_crawl_page = models.IntegerField(default=0)

    def __str__(self):
        return str(model_to_dict(self))

class Relationship(models.Model):
    related_review = models.ForeignKey('Review', on_delete=models.CASCADE)
    related_property = models.ForeignKey('Property', on_delete=models.CASCADE)
    best_sentence = models.CharField(max_length=1023, default='')
    prod = models.ForeignKey('Product', on_delete=models.CASCADE)
    sentiment = models.FloatField(default=0.0)
    clicked = models.IntegerField(default=0)
    rating_sum = models.IntegerField(default=0)
    rating_count = models.IntegerField(default=0)
    rating = models.FloatField(default=0.0)
    rank = models.FloatField(default=0.0)
    # added by sdk implementation
    url = models.CharField(max_length=1023, default='')
    # added by find_cluster
    subtopic = models.IntegerField(default=-1)

    def __str__(self):
        return str(model_to_dict(self))

    class Meta:
        unique_together = (("related_property", "related_review", "url"),)

class Review(models.Model):
    content = models.TextField(default='')
    prod = models.ForeignKey('Product', on_delete=models.CASCADE)
    review_id = models.CharField(max_length=1024, primary_key=True)

    # added by concurrency logic
    page = models.IntegerField(default=0)

    def __str__(self):
        return str(model_to_dict(self))

class Property(models.Model):
    prod = models.ForeignKey('Product', on_delete=models.CASCADE)
    xpath = models.CharField(max_length=1024, default='')
    topic = models.CharField(max_length=1024, default='')
    text_content = models.CharField(max_length=1024, default='')

    def __str__(self):
        return str(model_to_dict(self))
