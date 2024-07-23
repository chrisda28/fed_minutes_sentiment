import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime, date

CURRENT_YEAR = datetime.now().year   # scraping data of past 2 years
TWO_YEARS_AGO = CURRENT_YEAR - 2
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def get_fomc_urls():  # this function takes a couple of minutes to run
    """checks for FOMC minutes URLs over a two-year period,
     verifying each potential date and adding valid URLs to a list"""
    base_url = "https://www.federalreserve.gov/monetarypolicy/fomcminutes{}.htm"
    urls = []

    for year in range(TWO_YEARS_AGO, CURRENT_YEAR + 1):   # 2 year span
        for month in MONTHS:   # span all months
            for day in range(1, 31):  # span all possible days
                try:
                    meet_date = date(year, month, day)  # create date object
                    url = base_url.format(meet_date.strftime("%Y%m%d"))  # format the placeholder in URL above
                    response = requests.head(url)  # .head is used since not concerned with page content yet,
                    if response.status_code == 200:  # check if request went through
                        urls.append(url)  # attach valid url to list
                except ValueError:
                    continue  # its invalid so skip this iteration
    return urls


def scrape(url):
    date_string = url.split("/")[-1]   # get the data portion of the url
    date_string = date_string.replace(".htm", "")  # remove the .htm part to get just the date
    try:
        time.sleep(10)  # pause to not overload server
        response = requests.get(url)  # make request
        response.raise_for_status()  # error handling
        webpage = response.text
        soup = BeautifulSoup(webpage, 'html.parser')  # create scraping object
        article = soup.find('div', id='article')  # scraping entire article
        if not article:
            print(f"Could not find article with url: {url}")

        for item in article(["script", "style"]):  # removing JS and CSS elements in the article
            item.decompose()
            # Extract text and clean it
        lines = [line.strip() for line in article.get_text().splitlines() if line.strip()]
        text = '\n'.join(lines)
        return {"date": date_string, "text": text}  # return dict of data and article text
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def clean_text_files(scraped_data):
    """write FOMC minutes text into individual files"""
    for date, text in scraped_data.items():
        filename = f"{date}.txt"

        with open(file=filename, mode="w", encoding="utf-8") as file:
            file.write(text)
        print(f"Written: {filename}")


def main():
    urls = get_fomc_urls()
    scraped_data = {}
    for url in urls:
        result = scrape(url)
        if result:
            scraped_data[result['date']] = result['text']

    print("Writing data into files now")
    clean_text_files(scraped_data=scraped_data)
    print("Scraping complete.")


if __name__ == "__main__":
    main()


# def scrape(url):
#     date_string = url.split("/")[-1]   # get the data portion of the url
#     date_string = date_string.replace(".htm", "")  # remove the .htm part to get just the date
#     try:
#         time.sleep(10)  # pause to not overload server
#         response = requests.get(url)   # make request
#         if response.status_code == 200:  # check if request went through
#             webpage = response.text
#             soup = BeautifulSoup(webpage, 'html.parser')  # create scraping object
#             article = soup.find('div', id='article')  # scraping entire article
#
#             for item in article(["script", "style"]):  # removing JS and CSS elements in the article
#                 item.decompose()
#             text = article.get_text()   # removing HTML tags, leaving just the text
#             lines = [line.strip() for line in text.splitlines()]  # split text into lines and remove
#             # leading/trailing whitespace
#             chunks = []
#             for line in lines:  # break lines into phrases (chunks) and remove extra whitespace
#                 phrases = line.split("  ")   #
#                 for phrase in phrases:
#                     cleaned_phrase = phrase.strip()  # remove leading/trailing whitespace from each phrase
#                     chunks.append(cleaned_phrase)
#             cleaned_chunks = []
#             for chunk in chunks:  # filtering out empty chunks before appending
#                 if chunk:
#                     cleaned_chunks.append(chunk)
#             text = '\n'.join(cleaned_chunks)  # join clean chunks with new lines
#             return {"date": date_string, "text": text}  # return dict of data and article text
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return None
