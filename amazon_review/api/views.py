from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.exceptions import ObjectDoesNotExist
from django.forms import model_to_dict

from celery import shared_task, current_task, uuid
from celery.result import AsyncResult

from api.models import *
from api.tasks import *
from api.utils import parser, matcher, switcher, ranker

from datetime import date, timedelta
import time
import json


def index(request):
    return HttpResponse('success!\n')

def _success(stat, data):
    return JsonResponse({ 'stat': stat, **data })

def _fail(stat, error_msg):
    return JsonResponse({ 'stat': stat, 'error': error_msg })


# new version of prod() includes async logic
def prod(request):
    start_t = time.time()

    asin, start, cnt, url = "", 0, 0, ""
    try:
        asin, start, cnt, url = read_request_query(request)
    except ValueError as e:
        return _fail(400, "Error reading query: %s" % e)

    prod, properties = parse_product(asin)
    print("until product: ", time.time() - start_t)

    res = AsyncResult(asin)
    need = analyze_async_result(res, prod, url)

    # "PENDING" is default state for unknown tasks
    # so if "PENDING", means this "asin" has not been crawled
    # otherwise, the status would be "PROGRESS" or "SUCCESS"
    if res.state == "PENDING":
        prod.last_crawl_date = date.today()
        parse_async.apply_async((asin, 10, need, url), task_id=asin)

    # celery task keeps track of which page it has reached so far
    # wait until the desginated ending page number has been reached
    # if the crawler was called before, this while loop won't execute
    while res.state != "SUCCESS":
        if isinstance(res.info, Exception):
            return _fail(400, "parser raises: %s" % res.info)
        else:
            try:
                if res.info["page"] >= start + cnt: break
            except TypeError:
                pass
        # avoid querying task queue database too frequently
        time.sleep(0.1)
        # if time too long, fail no matter the status of crawler
        if time.time() - start_t > 60:
            return _fail(403, "parser takes too long to respond")

    print("until not blocking: ", time.time() - start_t)

    # has_more denotes whether there are more reviews
    # it equals false only if the following two conditions hold:
    # 1) there are no more relationships after the ending page,
    # 2) the crawler has already terminated
    more_rels = Relationship.objects.filter(
        prod=prod,
        url=url,
        related_review__page__gte=start+cnt,
    )
    has_more = (len(more_rels) > 0) or (res.state == "PROGRESS")

    # find relationshop has been rewritten for purpose of pagination
    data = find_relationship(prod, start, cnt, url)

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


# private implementations below
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


def read_request_query(request):
    query = request.GET.dict()

    if 'asin' not in query:
        raise ValueError("Query asin is required")
    asin = query["asin"]

    start = int(query["start"]) if "start" in query else 1
    cnt = int(query["count"]) if "count" in query else 2**31-1

    # url is empty if we use internal inference algorithm
    url = ""
    if request.method == "POST":
        post_data = json.loads(request.body.decode('utf-8'))
        if "url" not in post_data:
            raise ValueError("Cannot read URL from post data")
        url = post_data["url"]

    return asin, start, cnt, url


def parse_product(asin):
    # if product has already been crawled, then omit crawling again
    try:
        prod = Product.objects.get(pk=asin)
        properties = list(Property.objects.filter(prod=prod))
        return prod, properties
    except ObjectDoesNotExist:
        prod_query = parser.ParseProduct(asin)
        return save_prod(prod_query)


def analyze_async_result(res, prod, url):
    if res.state == "FAILURE":
        # if failed for custom URL reason, don't need to parse again
        known_errors = [
            "The response is not in a correct format",
            "The URL provided cannot be reached",
        ]
        ret = str(res.info) not in known_errors
        res.forget()
        return ret
    elif res.state == "SUCCESS":
        # if post request, cannot assume API behavior consistent
        if url != '':
            res.forget()
        # however, if already have relationships, no need to parse
        rels = Relationship.objects.filter(prod=prod)
        return len(rels) == 0
    return True


def find_relationship(prod, start, cnt, url):
    ret = { "payload": [] }
    related_properties = Property.objects.filter(prod = prod)

    for rp in related_properties:
        relationships = Relationship.objects.filter(
            prod=prod,
            url=url,
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
                    "clicked": int(content[4]),
                    "rating": float(content[5]), 
                    "subtopic": int(content[6])
                })

    return ret
