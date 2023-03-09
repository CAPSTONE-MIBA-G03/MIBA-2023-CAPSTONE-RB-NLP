# Imports
import time
from datetime import date
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup
from requests import Session


class Google:
    """
    A Class aimed at extracting relevant news article links from Google.

    Parameters
    ----------
    start_date : {}, default=None
        Specify a start date to be included in the Google search

    end_date : {}, default=None
        Specify a end date to be included in the Google search

    duration : {}, default=None
        Specify a specific number of months back from today in the Google search

    company : {}, default=None
        Specify a start date to be included in the Google search

    Examples
    --------
    >>> from link_extractor import Google
    >>> search = Google(company="Tesla")
    """

    ROOT = "https://www.google.com/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        #For very interesting reasons, the below headers actually make the requests fail or smth (returns empty df). Will have to implement some logging to debug!!!
        
        # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        # "Accept-Language": "en-US,en;q=0.9",
        # "Accept-Encoding": "gzip, deflate, br",
        # "Connection": "keep-alive",
        # "Referer": "https://www.google.com/",
    }

    def __init__(self, start_date=None, end_date=None, duration=None, company=None):
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.company = company

    def get_links(self, max_pages=None):
        """
        Extracts links from specified browser class and returns a dataframe object

        Parameters
        ----------
        max_pages : {}, default=None
            Specify the maximum number of google pages to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Browser" and "Link"

        Examples
        --------
        >>> from link_extractor import Google
        >>> search = Google(company="Tesla")
        >>> links = search.get_links(max_pages=5)
        """

        start_date = self.start_date
        end_date = self.end_date

        if start_date and end_date:
            date_range = f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
        else:
            date_range = ""

        search = f"{Google.ROOT}search?q={self.company}&hl=en&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjev5P_wqT9AhUZ7qQKHed-CvsQ_AUoAXoECAEQAw&biw=1245&bih=1046&dpr=2{date_range}"

        links = []
        page_count = 0

        with Session() as session:
            req = Request(search, headers=Google.HEADERS)
            page = urlopen(req).read()
            soup = BeautifulSoup(page, "lxml")

            while True:
                time.sleep(0.5)
                for item in soup.find_all("a", attrs={"class": "WlydOe"}):
                    link = item["href"]
                    links.append(link)

                # check if there are more pages to fetch
                next_page = soup.find("a", attrs={"id": "pnnext"})
                if next_page and (not max_pages or page_count < max_pages):
                    next_link = Google.ROOT + next_page.get("href")
                    req = Request(next_link, headers=Google.HEADERS)
                    page = urlopen(req).read()
                    soup = BeautifulSoup(page, "lxml")
                    page_count += 1
                else:
                    break

        fetched_links = pd.DataFrame({"Browser": "Google", "Link": links})
        return fetched_links


### TODO ###

# - Not sure if Class structure optimal
# - Instead of "max_pages", maybe "max_news"? So limit not pages, but links
# - ATM for testing a bit excessive use of "self." - should simplify
# - Make request session more robust by adding cookies, headers etc. Also, there is a way to make google think
#   we come from google.com (so we are not the whole time making direct requests to super specific urls) - not really solved, but a bit (maybe)
# - Spanish results??? Way around it?
