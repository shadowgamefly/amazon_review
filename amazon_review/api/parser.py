#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written as part of https://www.scrapehero.com/how-to-scrape-amazon-product-reviews-using-python/
from lxml import html, etree
import requests
import json,re
# from dateutil import parser as dateparser
from time import sleep
from utils import load_html

def ReviewURL(asin, page):
	return "https://www.amazon.com/product-reviews/" + asin + "/ref=cm_cr_arp_d_viewopt_srt?reviewerType=all_reviews&pageNumber=" + str(page) + "&sortBy=recent"

def request_parser(url, asin=None, page_count=None):
	html_page = load_html(url, asin, page_count)
	parser = html.fromstring(html_page)
	return parser

def is_blocked(parser):
	title = parser.xpath('//title/text()')
	# print(title)
	return title == 'Robot Check'
	# XPATH_CAPTCHA = '//form[@action="/errors/validateCaptcha"]'
	# has_captcha_form = parser.xpath(XPATH_CAPTCHA)


def ParseReviews(asin):
	# Added Retrying
	page_count = 0
	review_list = []
	while True:
		for i in range(5):
			try:
				#This script has only been tested with Amazon.com
				review_url = ReviewURL(asin, page_count)
				# Add some recent user agent to prevent amazon from blocking the request
				# Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
				parser = request_parser(review_url, asin, str(page_count))
				while(is_blocked(parser)):
					parser = request_parser(review_url, asin, str(page_count))

				reviews, count = parse_review_list(parser)
				if (count == 0):
					 return review_list
				review_list += reviews
				break
			except ValueError:
				print ("Retrying to get the correct response")
		page_count += 1

def ParseProduct(asin):
	for i in range(5):
		try:
			#This script has only been tested with Amazon.com
			amazon_url  = 'https://www.amazon.com/dp/'+asin
			# Add some recent user agent to prevent amazon from blocking the request
			# Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
			parser = request_parser(amazon_url, asin, '')
			general_info = parse_general(parser)
			property_dict = parse_property(parser)
			general_info.update({'asin': asin, 'properties':property_dict})
			return general_info
		except ValueError:
			print("Retrying to get the correct response")
	return {"error":"failed to process the page","asin":asin}

def parse_general(parser):
	XPATH_PRODUCT_NAME = '//h1//span[@id="productTitle"]//text()'
	raw_product_name = parser.xpath(XPATH_PRODUCT_NAME)
	product_name = ''.join(raw_product_name).strip()
	return {'name':product_name}
	# XPATH_AGGREGATE = '//span[@id="acrCustomerReviewText"]'
	# XPATH_AGGREGATE_RATING = '//table[@id="histogramTable"]//tr'
	# XPATH_PRODUCT_PRICE  = '//span[@id="priceblock_ourprice"]/text()'
	# grabing the rating  section in product page
	# ratings_dict = {}
	# raw_product_price = parser.xpath(XPATH_PRODUCT_PRICE)
	# product_price = ''.join(raw_product_price).replace(',','')
	# total_ratings  = parser.xpath(XPATH_AGGREGATE_RATING)
	# for ratings in total_ratings:
	# 	extracted_rating = ratings.xpath('./td//a//text()')
	# 	if extracted_rating:
	# 		rating_key = extracted_rating[0]
	# 		raw_raing_value = extracted_rating[1]
	# 		rating_value = raw_raing_value
	# 		if rating_key:
	# 			ratings_dict.update({rating_key:rating_value})

def parse_property(parser):
	XPATH_PRODUCT_TABLE_1 = '(//table[@id="productDetails_techSpec_section_1"]'
	XPATH_PRODUCT_TABLE_2 =	'(//table[@id="productDetails_techSpec_section_2"]'
	product_table_1 = parser.xpath(XPATH_PRODUCT_TABLE_1+'//th)')
	product_table_2 = parser.xpath(XPATH_PRODUCT_TABLE_2+'//th)')
	property_dict = {}
	for count, element in enumerate(product_table_1):
		property_dict[XPATH_PRODUCT_TABLE_1+'/tbody/tr/th)' + "[%d]"%(count + 1)] = element.xpath('./text()')[0].strip()
	for count, element in enumerate(product_table_2):
		property_dict[XPATH_PRODUCT_TABLE_2+'/tbody/tr/th)' + "[%d]"%(count + 1)] = element.xpath('./text()')[0].strip()
	# print(property_dict)
	return property_dict

def parse_review(parser):
	XPATH_REVIEW_SECTION_1 = '//div[contains(@id,"reviews-summary")]'
	XPATH_REVIEW_SECTION_2 = '//div[@data-hook="review"]'
	reviews = parser.xpath(XPATH_REVIEW_SECTION_1)
	if not reviews:
		reviews = parser.xpath(XPATH_REVIEW_SECTION_2)
	reviews_list = []
	if not reviews:
		raise ValueError('unable to find reviews in page')
	for review in reviews:
		review_dict = read_review_block(review)
		reviews_list.append(review_dict)
	return reviews_list

def parse_review_list(parser):
	XPATH_REVIEW_SECTION = '//div[@data-hook="review"]'
	reviews = parser.xpath(XPATH_REVIEW_SECTION)
	count = len(reviews)
	reviews_list = []
	for review in reviews:
		review_dict = read_review_block(review)
		reviews_list.append(review_dict)
	return reviews_list, count

def read_review_block(review):
	XPATH_RATING  = './/i[@data-hook="review-star-rating"]//text()'
	XPATH_REVIEW_HEADER = './/a[@data-hook="review-title"]//text()'
	XPATH_REVIEW_POSTED_DATE = './/a[contains(@href,"/profile/")]/parent::span/following-sibling::span/text()'
	XPATH_REVIEW_TEXT = './/span[@data-hook="review-body"]/node()'
	XPATH_AUTHOR  = './/a[contains(@href,"/profile/")]/parent::span//text()'
	raw_review_author = review.xpath(XPATH_AUTHOR)
	raw_review_rating = review.xpath(XPATH_RATING)
	raw_review_header = review.xpath(XPATH_REVIEW_HEADER)
	raw_review_posted_date = review.xpath(XPATH_REVIEW_POSTED_DATE)
	# raw_review_text = review.xpath('.//span[@data-hook="review-body"]/text()')

	raw_html_review_text = review.xpath(XPATH_REVIEW_TEXT)
	raw_review_text = []
	for i in range(len(raw_html_review_text)):
		if isinstance(raw_html_review_text[i], str):
			raw_review_text.append(raw_html_review_text[i])
		elif len(raw_review_text) > 0:
			if not raw_review_text[-1].endswith('.'):
				raw_review_text[-1] += '.'
	# print(raw_review_text)
	raw_review_id = review.get('id')
	author = ' '.join(' '.join(raw_review_author).split()).strip('By').strip()

	#cleaning data
	review_rating = ''.join(raw_review_rating).replace('out of 5 stars','')
	review_header = ' '.join(' '.join(raw_review_header).split())
	# print(raw_review_posted_date)
	# review_posted_date = dateparser.parse(''.join(raw_review_posted_date)).strftime('%d %b %Y')
	review_text = ' '.join(' '.join(raw_review_text).split())
	# print(review_text)
	#grabbing hidden comments if present
	review_dict = {
						'review_text':review_text,
						'review_id':raw_review_id,
						# 'review_posted_date':review_posted_date,
						'review_header':review_header,
						'review_rating':review_rating,
						'review_author':author,
					}
	return review_dict

def ReadAsin(asin):
	#Add your own ASINs here
	extracted_data = []
	print("Downloading and processing page https://www.amazon.com/dp/"+asin)
	extracted_data.append(ParseReviews(asin))
	return extracted_data

if __name__ == '__main__':
	print(ReadAsin('B01M18UZF5'))
