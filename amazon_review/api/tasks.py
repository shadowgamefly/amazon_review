from __future__ import absolute_import, unicode_literals
from django.db.utils import IntegrityError

from celery import shared_task, current_task, group
from celery.result import AsyncResult, allow_join_result

from api.models import *
from api.utils import parser, matcher

import time, requests, json


@shared_task
def parse_async(asin, num_workers=1, need_parse=True, url=''):
    # set the state != "PENDING", to notify that the task has started
    current_task.update_state(state="PROGRESS", meta={"page": 1})

    # celery only accepts json serializable objects as parameters
    # so we cannot pass "prod", "properties" directly
    # instead, we have to query the database again for these objects
    prod = Product.objects.get(pk=asin)
    properties = list(Property.objects.filter(prod=prod))

    pc = 1
    while True:
        # variable indicating whether the process needs to break
        need_break = False
        if need_parse:
            gp = group(worker.s(asin, pc+w) for w in range(num_workers))
            res = gp.apply_async()

            ret_list = None
            with allow_join_result():
                try:
                    ret_list = res.get(timeout=10)
                    need_break = correct_terminate(ret_list)
                except Exception as e:
                    raise ValueError("%s in worker" % e)

            reviews = [r for x in ret_list for r in x[0]]
            save_review(reviews, prod, pc)
        else:
            tmp = Review.objects.filter(page__gte=pc+num_workers)
            need_break = (len(tmp) == 0)

        saved_reviews = list(Review.objects.filter(
            page__gte=pc, page__lt=pc+num_workers,
        ))
        property_data = [convert_property(p) for p in properties]
        review_data = [convert_review(r) for r in saved_reviews]

        rels = []
        if url == "":
            rels = matcher.htmm_inference(property_data, review_data)
        else:
            rels = fetch_from_api(url, property_data, review_data)
        save_relationship(rels, prod, url)

        # update task state to reflect the increment in page count
        # "meta" can be accessed in main process as "AsyncResult.info"
        # which is not pointed out clearly in celery documentation
        if need_break: break
        pc += num_workers
        current_task.update_state(state="PROGRESS", meta={"page": pc})


@shared_task
def worker(asin, pc):
    for i in range(5):
        try:
            review_page = parser.ReviewURL(asin, pc)
            rp = parser.request_parser(review_page, asin, pc)
            retry = 0
            while parser.is_blocked(rp):
                rp = parser.request_parser(review_page, asin, pc)
                retry += 1
                if retry >= 10: raise ValueError("Too many robot checks")

            reviews, count = parser.parse_review_list(rp)
            print("asin: %s, page: %d, count: %d" % (asin, pc, count))
            return (reviews, pc)

        except ValueError as e:
            print(e, "; retrying max 5 times")

    return ([], pc)


# private implementations below
def save_review(reviews, prod, page_count):
    for review_info in reviews:
        try:
            review = Review.objects.create(
                review_id=review_info['review_id'],
                content=review_info['review_text'],
                prod=prod,
                page=page_count,
            )
        except IntegrityError:
            pass


def save_relationship(relationships, prod, url):
    for relation in relationships:
        relation['prod'] = prod
        relation['url'] = url
        try:
            Relationship.objects.create(**relation)
        except IntegrityError:
            pass


def convert_property(p):
    return {
        "id": p.id,
        "prod": p.prod.asin,
        "topic": p.topic,
        "text_content": p.text_content,
        "xpath": p.xpath,
    }


def convert_review(r):
    return {
        "content": r.content,
        "id": r.review_id,
        "prod": r.prod.asin,
        "page": r.page,
    }


def correct_terminate(ret_list):
    sorted_l = sorted(ret_list, key=lambda tup: tup[1])
    for i in range(len(sorted_l)):
        if len(sorted_l[i][0]) < 10 and i < len(sorted_l)-1:
            for j in range(i+1, len(sorted_l)):
                if len(sorted_l[j][0]) != 0: return False
            return True
    return False


def fetch_from_api(url, property_data, review_data):
    data = {"properties": property_data, "reviews": review_data}
    post_data = json.dumps(data, indent=2)
    headers = {"content-type": "application/json"}

    try:
        resp = requests.post(url, data=post_data, headers=headers)
        rel_data = resp.json()
        return rel_data["relationships"]
    except ConnectionError:
        raise ValueError("The URL provided cannot be reached")
    except ValueError:
        raise ValueError("The response is not in a correct format")
