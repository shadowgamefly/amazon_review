# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2018-01-26 17:14
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_auto_20180125_2118'),
    ]

    operations = [
        migrations.AddField(
            model_name='relationship',
            name='rating',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='relationship',
            name='rating_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='relationship',
            name='rating_sum',
            field=models.IntegerField(default=0),
        ),
    ]