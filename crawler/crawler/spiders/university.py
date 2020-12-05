import pathlib
import nltk
from tinydb import TinyDB
from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from pathlib import Path
import uuid


class UniversitySpider(CrawlSpider):
    name = 'university'
    allowed_domains = [
        'umd.edu',
        'stanford.edu',
        'cmu.edu'
    ]
    start_urls = [
        'https://www.umd.edu/',
        'https://www.stanford.edu/',
        'https://www.cmu.edu/'
    ]
    base_url = 'https://www.umd.edu/'
    rules = [Rule(LinkExtractor(deny=r'.+?(Patapov|event|publications|labs).+?', unique=True),
                  callback='parse', follow=True)]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._db = TinyDB('pages.json')
        self._pages = pathlib.PurePath(pathlib.Path().absolute(),"pages")
        Path(self._pages).mkdir(parents=True, exist_ok=True)

    def parse(self, response):
        text = BeautifulSoup(response.xpath('//*').get()).get_text()
        tokens = nltk.word_tokenize(text)
        normalized_text = ' '.join([word for word in tokens if word.isalnum()])
        file_path = pathlib.PurePath(self._pages, "{}.html".format(uuid.uuid1()))
        self._db.insert({"url": response.request.url, "file": file_path.as_uri()})
        with open(file_path, 'w') as f:
            f.write(normalized_text)