from django.db import models

class Product(models.Model):
    title = models.CharField(max_length=1024, default='')
    prod_id = models.CharField(max_length=1024, default='', primary_key=True)
    date = models.DateField(default=datetime.date.today)

class User(models.Model):
    username = models.CharField(max_length=1024, default='')
    user_id = models.CharField(max_length=1024, default='', primary_key=True)

class Review(models.Model):
    content = models.TextField(default='')
    prod = models.ForeignKey('Product', on_delete=models.CASCADE)
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    review_id = models.CharField(max_length=1024, primary_key=True)

class Property(models.Model):
    prod = models.ForeignKey('Product', on_delete=models.CASCADE)
    xpath = models.CharField(max_length=1024, default='')
    text_content = models.CharField(max_length=1024, default='')