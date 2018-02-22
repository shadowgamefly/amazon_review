from __future__ import absolute_import, unicode_literals
from api.models import *
import time, math

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer

from .topWords import topics

def keyword_match(properties, reviews, prod):
    # print(properties)
    # print(reviews)
    sid = SentimentIntensityAnalyzer()
    ret = []
    start_time = time.time()
    for property in properties:
        text = property.text_content;
        property_word = [" " + word.lower() + " " for word in text.split(" ")]
        for review in reviews:
            if not isinstance(review.content, str): continue
            sentences = tokenize.sent_tokenize(review.content)
            # print(sentences)
            dct = {}
            maxCount = 0
            best_sentence = ''
            for i in range(len(sentences)):
                count = sum(1 for word in property_word if word in sentences[i].lower())
                if count > maxCount:
                    maxCount = count
                    best_sentence = sentences[i]
            if maxCount >= 5:
                ss = sid.polarity_scores(best_sentence.lower())['compound']
                ret.append({'related_property': property, 'best_sentence': best_sentence, 'related_review': review, 'sentiment': ss})
    save_relationship(ret, prod)
    # elapsed_time = time.time() - start_time
    # print(elapsed_time)
    return ret

def keyword_match_gibbs(properties, reviews, prod):
    sid = SentimentIntensityAnalyzer()
    twt = TreebankWordTokenizer()
    ss = SnowballStemmer('english')
    ret = []

    start_time = time.time()

    for review in reviews:
        if not isinstance(review.content, str): continue
        sentences = tokenize.sent_tokenize(review.content)

        for sentence in sentences:
            tmp = [ss.stem(w.lower()) for w in twt.tokenize(sentence)]
            words = ["NUM" if w.isdigit() else w for w in tmp]
            for p in properties:
                score = 0
                for w in words:
                    if p.topic in topics and w in topics[p.topic]:
                        score += math.log(topics[p.topic][w])
                perplexity = math.exp(-score / len(words))
                # print(perplexity)
                if perplexity > 3.0:
                    ps = sid.polarity_scores(sentence.lower())['compound']
                    ret.append({'related_property': p, 'best_sentence': sentence, 'related_review': review, 'sentiment': ps})

    save_relationship(ret, prod)

    elapsed_time = time.time() - start_time
    print(elapsed_time)

    return ret

def save_relationship(relationships, prod):
    for relation in relationships:
        # print(relation)
        relation['prod'] = prod
        rel = Relationship(**relation)
        rel.save()
