import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime, date

CURRENT_YEAR = datetime.now().year   # scraping data of past 2 years
TWO_YEARS_AGO = CURRENT_YEAR - 2
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def get_fomc_urls():  # this function takes a couple of minutes to run
    """return list of valid fomc minutes url for past 2 years"""
    base_url = "https://www.federalreserve.gov/monetarypolicy/fomcminutes{}.htm"
    urls = []

    for year in range(TWO_YEARS_AGO, CURRENT_YEAR + 1):
        for month in MONTHS:
            for day in range(1, 31):
                try:
                    meet_date = date(year, month, day)
                    url = base_url.format(meet_date.strftime("%Y%m%d"))
                    response = requests.head(url)  # .head is used since not concerned with page content yet
                    if response.status_code == 200:  # check if request went through
                        urls.append(url)  # attach valid url to list
                except ValueError:
                    continue  # its invalid so skip this iteration
    return urls

# testurl = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20220126.htm"
# response = requests.get(testurl)
# webpage = response.text
# soup = BeautifulSoup(webpage, 'html.parser')
# article = soup.find('div', id='article')
# for script in article(["script", "style"]):
#     script.decompose()
# text = article.get_text()
# # lines = (line.strip() for line in text.splitlines())
# #
# # # Break multi-headlines into a line each
# # chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# #
# # # Drop blank lines
# # text = '\n'.join(chunk for chunk in chunks if chunk)
# print(text)


def scrape(url):
    date_string = url.split("/")[-1]   # get the data portion of the url
    date_string = date_string.replace(".htm", "")  # remove the .htm part of the url
    try:
        time.sleep(10)
        response = requests.get(url)
        if response.status_code == 200:
            webpage = response.text
            soup = BeautifulSoup(webpage, 'html.parser')
            article = soup.find('div', id='article')
            return {"date": date_string, "text": article.get_text()}  # return dict of data and article text
    except Exception as e:
        print(f"error {str(e)}")
        return None


# def clean_text(text):
#     count = 1
#     for item in article(["script", "style"]):
#         item.decompose()
#     article_text = article.get_text()
#     with open(file=f"file{count}", mode="w") as file:
#         csv_file = file.write(article_text)
#
#     count += 1





