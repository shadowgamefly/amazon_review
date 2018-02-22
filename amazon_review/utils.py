import requests, random
from credential import proxies

headers = [
	{'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'},
	{'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:58.0) Gecko/20100101 Firefox/58.0'},
	{'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36'},
	{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/13.10586'},
	{'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64, x64; Trident/7.0; rv:11.0) like Gecko'},
]

def write_html(data, filename='sample.html'):
	with open(filename, 'w', encoding='utf-8') as f:
		f.write(data.content.decode('utf-8'))

def select_headers():
	idx = random.randint(0, len(headers)-1)
	return headers[idx], idx

def select_proxies():
	idx = random.randint(0, len(proxies)-1)
	return proxies[idx], idx

def load_html(url, asin, pageid):
	header, hid = select_headers()
	proxy, pid = select_proxies()
	page = requests.get(url, headers = header, proxies=proxy)

	filename = str("%s_%s_%s_%s"%(asin, pageid, str(hid), str(pid)))
	# write_html(page, "/af12/jw7jb/public_html/%s.html"%(filename))

	page_response = page.text
	return page_response

if __name__ == '__main__':
	print(select_headers())
	print(select_headers())
