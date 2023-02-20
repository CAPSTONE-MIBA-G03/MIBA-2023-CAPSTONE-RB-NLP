# Imports
from datetime import date
from urllib.request import Request, urlopen

import pandas as pd
import requests
from bs4 import BeautifulSoup


class Google:
    """
    A Class aimed at extracting relevant news article links from Google
    """

    ROOT = "https://www.google.com/"

    def __init__(
        self,
        start_date=None, #date.today().strftime("%d/%m/%Y"),
        end_date=None,
        duration=None,
        company=None
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.company = company

    def get_links(self, pages="all"):
        """
        Extracts links from specified browser class and returns a dataframe object
        """
        self.pages = pages

        start_date = self.start_date
        end_date = self.end_date
        if start_date and end_date:
            date_range = f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
        else:
            date_range = ""

        search = f"{Google.ROOT}search?q={self.company}&hl=en&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjev5P_wqT9AhUZ7qQKHed-CvsQ_AUoAXoECAEQAw&biw=1245&bih=1046&dpr=2{date_range}"
        req = Request(search, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"})
        page = urlopen(req).read()
        links = []

        with requests.Session() as session:
            soup = BeautifulSoup(page, "lxml")
            for item in soup.find_all("a", attrs={"class": "WlydOe"}):
                link = item["href"]
                links.append(link)

        fetched_links = pd.DataFrame({"Browser": "Google", "Link": links})
        return fetched_links

            #next_page = soup.find('a', attrs={'id':'pnnext'}) -> This is already the correct "next page" substring (ROOT + next_page)
            # just gotta implement looping logic/maybea as method input how many pages? easier testing.
            
            # next E (next ['href'])
            # link = root + next
            # news (link)


tesla = Google(company="Tesla")
print(tesla.get_links())


### TODO ###

# - 
# - Not sure if Class structure optimal
# - Implement search over multiple pages
# - ATM for testing a bit excessive use of "self." - should simplify
# - Make request session more robust by adding cookies, headers etc. Also, there is a way to make google think
#   we come from google.com (so we are not the whole time making direct requests to super specific urls)