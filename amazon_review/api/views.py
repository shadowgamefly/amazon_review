from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.exceptions import ObjectDoesNotExist
from django.forms import model_to_dict

from celery import shared_task, current_task, uuid
from celery.result import AsyncResult

from api.models import *
from api.tasks import *
from . import parser, matcher, switcher, ranker

from datetime import date, timedelta
import time


def index(request):
    return HttpResponse('success!\n')

def _success(stat, data):
    return JsonResponse({ 'stat': stat, **data })

def _fail(stat, error_msg):
    return JsonResponse({ 'stat': stat, 'error': error_msg })


# new version of prod() includes async logic
def prod(request):
    start_t = time.time()

    query = request.GET.dict()
    asin = query["asin"]
    start = int(query["start"]) if "start" in query else 1
    cnt = int(query["count"]) if "count" in query else 2**31-1

    # if product has already been crawled, then omit crawling again
    prod = None; properties = []
    try:
        prod = Product.objects.get(pk=asin)
        properties = list(Property.objects.filter(prod=prod))
    except ObjectDoesNotExist:
        prod_query = parser.ParseProduct(asin)
        prod, properties = save_prod(prod_query)

    # measure time until crawling product
    print("until product: ", time.time() - start_t)

    # if the last crawl date is older than a week, then crawl again
    res = AsyncResult(asin)
    if date.today() - prod.last_crawl_date > timedelta(7):
        if res.state == "SUCCESS": res.forget()

    # query celery task queue for the crawler process
    # "PENDING" is default state for unknown tasks
    # so if "PENDING", means this "asin" has not been crawled
    # otherwise, the status would be "PROGRESS" or "SUCCESS"
    if res.state == "PENDING":
        prod.last_crawl_date = date.today()
        parse_async.apply_async((asin, 10), task_id=asin)

    # celery task keeps track of which page it has reached so far
    # wait until the desginated ending page number has been reached
    # if the crawler was called before, this while loop won't execute
    while res.state != "SUCCESS":
        # if res.info is None: continue
        try:
            if res.info["page"] >= start+cnt: break
        except:
            continue
        # avoid querying task queue database too frequently
        time.sleep(0.1)
        # print("inside while loop")

    # measure time until the (blocking) wait expires
    print("until not blocking: ", time.time() - start_t)

    # has_more denotes whether there are more reviews
    # it equals false only if the following two conditions hold:
    # 1) there are no more relationships after the ending page,
    # 2) the crawler has already terminated
    more_rels = Relationship.objects.filter(
        prod=prod,
        related_review__page__gte=start+cnt,
    )
    has_more = (len(more_rels) > 0) or (res.state == "PROGRESS")

    # find relationshop has been rewritten for purpose of pagination
    data = find_relationship(prod, start, cnt)

    print("total: ", time.time() - start_t)
    return _success(200, { "has_more": has_more, **data })


def click(request):
    query = request.GET.dict()
    relationship_id = query['id']
    try:
        relationship = Relationship.objects.get(pk=relationship_id)
        relationship.clicked += 1
        relationship.save()
        return _success(200, model_to_dict(relationship))
    except ObjectDoesNotExist:
        return _fail(404, "Relationship not found")


def rate(request):
    query = request.GET.dict()
    relationship_id = query['id']
    score = query['rating']
    try:
        relationship = Relationship.objects.get(pk=relationship_id)
        relationship.rating_sum += int(score)
        relationship.rating_count += 1
        relationship.rating = relationship.rating_sum/relationship.rating_count
        relationship.save()
        return _success(200, model_to_dict(relationship))
    except ObjectDoesNotExist:
        return _fail(404, "Relationship not found")


# private implementations
def save_prod(query):
    asin = query['asin']
    product_name = query['name']
    try:
        prod = Product.objects.get(pk=asin)
        properties = list(Property.objects.filter(prod=prod))
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
        # print(switch.switch(value))
        property = Property.objects.create(xpath = key, prod = prod, text_content = switcher.switch(value), topic = value)
        property.save()
        saved_properties.append(property)
    return saved_properties

def find_relationship(prod, start, cnt):
    ret = { "payload": [] }
    related_properties = Property.objects.filter(prod = prod)
    for rp in related_properties:
        relationships = Relationship.objects.filter(
            prod=prod,
            related_property=rp,
            related_review__page__gte=start,
            related_review__page__lt=start+cnt,
        )
        ret["payload"].append({
            "xpath": rp.xpath, "topic": rp.topic, "reviews": [],
        })
        for value in ranker.rank(relationships):
            for best_sentence, content in value.items():
                ret["payload"][-1]["reviews"].append({
                    "content": best_sentence,
                    "id": content[1],
                    "sentiment": float(content[2]),
                    "ranking": int(content[3]),
                })
    return ret

####Deprecated####
# def highlight(request):
#     query = request.GET.dict()
#     review_id = query['review']
#     related_review = Review.objects.get(pk=review_id)
#     asin = related_review.prod.asin
#     related_prod = Product.objects.get(pk=asin)
#     all_relation = Relationship.objects.filter(related_review=related_review, prod=related_prod)
#     ret_list = []
#     for relation in all_relation:
#         ret_list.append({'sentence':relation.best_sentence,
#         'property' : relation.related_property.topic,
#         'sentiment' : relation.sentiment})
#     return _success(200, {'content':ret_list})
