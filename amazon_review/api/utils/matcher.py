from __future__ import absolute_import, unicode_literals
from api.models import *
import time, math, sys, uuid

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer

from pyhtmm.utils import load_pickle
from pyhtmm.process import process_doc

sid = SentimentIntensityAnalyzer()
htmm = load_pickle('./data/htmm_trained_model.pickle')
word2index = load_pickle('./data/word2index.pickle')

idx_topic = [
    "Screen Size",
    "Graphics Coprocessor",
    "Processor",
    "RAM",
    "Operating System",
    "Hard Drive",
    "Number of USB 3.0 Ports",
    "Average Battery Life (in hours)",
]

import gensim, json, pickle, re, torch
import numpy as np

model = torch.load('./data/infersent.allnli.pickle')
cluster_centers = pickle.load(open('./data/centers.pickle', 'rb'))
model.set_glove_path('./data/glove.840B.300d.txt')
model.build_vocab_k_words(K=100000)

def keyword_match(properties, reviews):
    ret = []

    for p in properties:
        text = p["text_content"]
        p_word = [" " + word.lower() + " " for word in text.split(" ")]
        for review in reviews:
            if not isinstance(review["content"], str): continue
            sentences = tokenize.sent_tokenize(review["content"])
            dct = {}
            maxCount = 0
            best_sentence = ''
            for i in range(len(sentences)):
                count = sum(1 for w in p_word if w in sentences[i].lower())
                if count > maxCount:
                    maxCount = count
                    best_sentence = sentences[i]
            if maxCount >= 5:
                ps = sid.polarity_scores(best_sentence.lower())['compound']
                ret.append({
                    'related_property_id': p["id"],
                    'best_sentence': best_sentence,
                    'related_review_id': review["id"],
                    'sentiment': ps,
                    'htmm_topic': 0,
                })

    find_cluster(ret)
    return ret

def htmm_inference(properties, reviews):
    ret = []

    idx_id = [""] * len(idx_topic)
    for p in properties:
        for i in range(len(idx_topic)):
            if idx_topic[i] == p["topic"]: idx_id[i] = p["id"]

    for review in reviews:
        doc = process_doc(review["content"], word2index)
        path, entropy = htmm.predict_topic(doc)
        for i, stn in enumerate(doc.sentence_list):
            if entropy[i] > 0.01 or stn.num_words < 5: continue
            ps = sid.polarity_scores(stn.raw_content.lower())['compound']
            ret.append({
                'related_property_id': idx_id[path[i] % 8],
                'best_sentence': stn.raw_content,
                'related_review_id': review["id"],
                'sentiment': ps,
                'htmm_topic': path[i] % 8,
            })

    find_cluster(ret)
    return ret

def find_cluster(relationships):
    margin = 40 ## Need testing
    text_content = [relationship['best_sentence'] for relationship in relationships]
    embeded_content = model.encode(text_content, bsize=128, tokenize=True, verbose=True)
    
    for sent_id, relationship in enumerate(relationships):
        closest_cluster = -1
        closest_dist = float('inf')
        for cluster_id, cluster in enumerate(cluster_centers[str(relationship['htmm_topic'])]):
            dist = np.linalg.norm(cluster - embeded_content[sent_id])
            if closest_dist > dist and dist < margin :
                 closest_cluster = cluster_id
                 closest_dist = dist
        relationship["subtopic"] = closest_cluster
        relationship.pop('htmm_topic', None)
