
# importing libraries
import pandas as pd
import scrapy
from scrapy import signals
from scrapy.xlib.pydispatch import dispatcher
from urllib.parse import urlencode
from decimal import Decimal
from re import sub


# to crawl use command scrapy crawl Wikipedia_movies_budget_spider
class wikiMoviesSpider(scrapy.Spider):
    name = "Wikipedia_movies_budget_spider"
    wikipediaquerydict = {}
    filename = 'movies-production-budget.txt'
    movieData = []
    def __init__(self):
        dispatcher.connect(self.engine_stopped, signals.engine_stopped)

    # get movie id and names from excel file
    # then add the data to a dict
    def start_requests(self):
        # importing the dataset
        dataset = pd.read_excel('Scoring Sheet_wikiurl.xlsx')
        # X starts from productio_year and omits total
        self.movieData = dataset.iloc[:, 0:4].values
        # print(self.movieData)
        # print(len(self.movieData))


        for movieDataItem in self.movieData:
            self.wikipediaquerydict[movieDataItem[0]] = movieDataItem[3]
        # print(self.wikipediaquerydict)
        for id in self.wikipediaquerydict:
            if self.wikipediaquerydict[id] != 0:
                yield scrapy.Request(url= self.wikipediaquerydict[id], callback=self.parse)
                # print(self.wikipediaquerydict[id])
                # break

    # parse and replace dict values with either the new url or with 0
    def parse(self, response):
        # page = response.url.split("/")[-2]

        headerarr = response.selector.xpath('//table[@class="infobox vevent"]/tr/th/text()').extract()
        dataarr = response.selector.xpath('//table[@class="infobox vevent"]/tr/td/text()').extract()
        budget_found = False
        budget_position_from_end = 0
        for header in reversed(headerarr):
            if header.lower() == 'budget':
                budget_found = True
                budget_position_from_end = list(reversed(headerarr)).index(header)
        # print(response.request.url)
        if budget_found:
            for id in self.wikipediaquerydict:
                if response.request.url == self.wikipediaquerydict[id]:
                    mon = list(reversed(dataarr))[budget_position_from_end].strip()
                    # code to clean up money
                    oldmon = mon
                    mon = mon.replace('–', '-')
                    dashloc = mon.find('-')
                    # print(dashloc)
                    if dashloc > 0:
                        mon = mon[0:dashloc]

                    value = Decimal(sub(r'[^-?\d.]', '', mon))
                    if 'million' in oldmon.lower():
                        value = value * 10 ** 6
                    # end code to clean up money
                    self.wikipediaquerydict[id] = value
                    break
        else:
            for id in self.wikipediaquerydict:
                if response.request.url == self.wikipediaquerydict[id]:
                    self.wikipediaquerydict[id] = '0'
                    self.log('Url not found')
                    break

    def engine_stopped(self):
        with open(self.filename, 'a+') as f:
            for movieDataItem in self.movieData:
                f.write(str(self.wikipediaquerydict[movieDataItem[0]]))
                f.write('\n')
