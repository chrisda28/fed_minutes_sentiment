from bs4 import BeautifulSoup
import requests
from datetime import datetime, date

CURRENT_YEAR = datetime.now().year   # scraping data of past 2 years
TWO_YEARS_AGO = CURRENT_YEAR - 2
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# response = requests.get(urlhere)
# webpage = response.text
# soup = BeautifulSoup(webpage, 'html.parser')


def get_fomc_urls():
    """return list of valid fomc minutes url for past 2 years"""
    base_url = "https://www.federalreserve.gov/monetarypolicy/fomcminutes{}.htm"
    urls = []

    for year in range(TWO_YEARS_AGO, CURRENT_YEAR + 1):
        for month in MONTHS:
            for day in range(1, 31):
                try:
                    meet_date = date(year, month, day)
                    url = base_url.format(meet_date.strftime("%Y%m%d"))
                    response = requests.head(url)
                    if response.status_code == 200:  # check if request went through
                        urls.append(url)
                except ValueError:
                    continue  # its invalid so skip this iteration
    return urls

hi = get_fomc_urls()
print(hi)