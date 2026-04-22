#src/loaders/web_loader.py

import requests
from bs4 import BeautifulSoup

def load_webpage(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.get_text()