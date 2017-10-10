from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.exceptions import ObjectDoesNotExist
from api.models import *
from django.forms import model_to_dict
from . import parser, match

def index(request):
    return HttpResponse('success!\n')

def _success(stat, data):
    return JsonResponse({ 'stat': stat, **data })

def _fail(stat, error_msg):
    return JsonResponse({ 'stat': stat, 'error': error_msg })

def index(request):
    return HttpResponse("Hello World!")

def prod(request):
    query = request.GET.dict()
    asin = query['asin']
    prod, properties, reviews = parse(asin)
    relationships = match.keyword_match(properties, reviews)
    save_relationship(relationships, prod)
    ret = find_relationship(asin)
    return _success(200, ret)

def parse(asin):
    reviews = parser.ParseReviews(asin)
    prod_query = parser.ParseProduct(asin)
    prod, properties = save_prod(prod_query)
    saved_reviews = save_review(reviews, prod)
    return prod, properties, saved_reviews

def find_relationship(prod):
    ret = {}
    related_properties = Property.objects.filter(prod = prod)
    for property in related_properties:
        relationships = Relationship.objects.filter(prod = prod, related_property = property)
        ret[property.xpath] = []
        for relationship in relationships:
            review = relationship.related_review
            ret[property.xpath].append({relationship.best_sentence: [review.content, review.review_id]})
    return ret

def save_relationship(relationships, prod):
    for relation in relationships:
        relation['prod'] = prod
        rel = Relationship(**relation)
        rel.save()

def save_prod(query):
    asin = query['asin']
    product_name = query['name']
    try:
        prod = Product.objects.get(pk=asin)
        properties = Property.objects.filter(prod=prod)
    except ObjectDoesNotExist:
        prod = Product.objects.create(asin=asin, title=product_name)
        prod.save()
        properties = save_property(query, prod)
    return prod, properties

def save_review(reviews, prod):
    saved_reviews = []
    for review_info in reviews:
        try:
            review = Review.objects.get(review_id = review_info['review_id'])
        except ObjectDoesNotExist:
            review = Review.objects.create(review_id = review_info['review_id'], content=review_info['review_text'], prod=prod)
            saved_reviews.append(review)
    return saved_reviews

def save_property(query, prod):
    properties = query['properties']
    saved_properties = []
    for key, value in properties.items():
        property = Property.objects.create(xpath = key, prod = prod, text_content = value)
        property.save()
        saved_properties.append(property)
    return saved_properties
