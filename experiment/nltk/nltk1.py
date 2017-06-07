import nltk
import re
import pprint
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup

url = "http://sohoa.vnexpress.net/tin-tuc/cong-dong/hoi-dap/chon-ong-dong-loai-nao-cho-dieu-hoa-de-tiet-kiem-dien-3594890.html"
# response = request.urlopen(url)
# raw = response.read().decode('utf8')
html = request.urlopen(url).read().decode('utf8')
raw = BeautifulSoup(html, "lxml").get_text()
print("content length:", len(raw))
tokens = word_tokenize(raw)
# print(tokens[:10])
text = nltk.Text(tokens)
# print(text[1000:1020])
print(text.collocations())
words = [w.lower() for w in text]
vocab = sorted(set(words))
print(vocab)