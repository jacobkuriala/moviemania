
# importing libraries
import pandas as pd
import scrapy
from scrapy import signals
from scrapy.xlib.pydispatch import dispatcher
from urllib.parse import urlencode


# to crawl use command scrapy crawl Wikipedia_movies_url_spider
class wikiMoviesSpider(scrapy.Spider):
    name = "Wikipedia_movies_url_spider"
    wikipediaquerydict = {}
    filename = 'movies-url.txt'
    movieData = []
    def __init__(self):
        dispatcher.connect(self.engine_stopped, signals.engine_stopped)

    # get movie id and names from excel file
    # then add the data to a dict
    def start_requests(self):
        # importing the dataset
        dataset = pd.read_excel('Scoring Sheet.xlsx')
        # X starts from productio_year and omits total
        self.movieData = dataset.iloc[:, 0:4].values
        # print(self.movieData)
        # print(len(self.movieData))


        for movieDataItem in self.movieData:
            self.wikipediaquerydict[movieDataItem[0]] = 'https://en.wikipedia.org/w/index.php?' + \
                                      urlencode({'search': 'movie ' + str(movieDataItem[2]) + ' ' + str(movieDataItem[3])})
        # print(self.wikipediaquerydict)
        for id in self.wikipediaquerydict:
            yield scrapy.Request(url= self.wikipediaquerydict[id], callback=self.parse)

    # parse and replace dict values with either the new url or with 0
    def parse(self, response):
        # page = response.url.split("/")[-2]

        temparr = response.selector.xpath(
            '//ul[@class="mw-search-results"]/li/div[@class="mw-search-result-heading"]/a/@href').extract()
        if (len(temparr) > 0):
            for id in self.wikipediaquerydict:
                if response.url == self.wikipediaquerydict[id]:
                    self.wikipediaquerydict[id] = 'https://en.wikipedia.org' + temparr[0]
                    break
        else:
            for id in self.wikipediaquerydict:
                if response.url == self.wikipediaquerydict[id]:
                    self.wikipediaquerydict[id] = '0'
                    self.log('Url not found')
                    break

    def engine_stopped(self):
        with open(self.filename, 'a+') as f:
            for movieDataItem in self.movieData:
                f.write(self.wikipediaquerydict[movieDataItem[0]])
                f.write('\n')

