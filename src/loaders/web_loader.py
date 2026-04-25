import requests
from bs4 import BeautifulSoup


def load_webpage(url: str):

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            print("HTTP ERROR:", r.status_code)
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        text = " ".join(text.split())

        return text

    except Exception as e:
        print("SCRAPE ERROR:", str(e))
        return ""