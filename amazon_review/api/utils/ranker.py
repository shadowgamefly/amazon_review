from api.models import *
from django.forms import model_to_dict

def rank(relationships):
    return rank_by_click(relationships)

def present(ordered_relationships):
    relationship_list = []
    for relationship in ordered_relationships:
        review = relationship.related_review
        relationship_list.append({relationship.best_sentence: [review.content, review.review_id, relationship.sentiment, relationship.pk, relationship.clicked, relationship.rating, relationship.subtopic]})
    return relationship_list

def rank_by_click(relationships):
    return present(sorted(relationships, key=lambda k: k.clicked, reverse=True))

def no_rank(relationships):
    return present(relationships)
