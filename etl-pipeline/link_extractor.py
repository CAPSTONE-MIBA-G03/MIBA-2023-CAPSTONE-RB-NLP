# Imports
import asyncio
import time
from datetime import date
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
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
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }

    def __init__(
        self, start_date=None, end_date=None, duration=None, company=None, country="us"
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.company = company
        self.country = country

    def get_links(self, max_articles=None) -> pd.DataFrame:
        """
        Extracts links from specified browser class and returns a dataframe object

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of google articles to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Browser" and "Link"

        Examples
        --------
        >>> from link_extractor import Google
        >>> search = Google(company="Tesla")
        >>> links = search.get_links(max_articles=50)
        """

        # adding date query parameters
        if self.start_date and self.end_date:
            date_range = f"&tbs=cdr:1,cd_min:{self.start_date},cd_max:{self.end_date}"
        else:
            date_range = ""

        # define full search url
        search = f"{Google.ROOT}search?q={self.company}&hl=en&source=lnms&tbm=nws&gl={self.country}&sa=X&ved=2ahUKEwjev5P_wqT9AhUZ7qQKHed-CvsQ_AUoAXoECAEQAw&biw=1245&bih=1046&dpr=2{date_range}"

        # define loop variables
        links = []
        article_count = 0

        # create session and extract links from different pages
        with Session() as session:
            req = Request(search, headers=Google.HEADERS)
            page = urlopen(req).read()
            soup = BeautifulSoup(page, "lxml")

            while True:

                # extract HTML anchors in the page
                anchors = soup.select("#search a")

                # append each of the "href" to the links
                for a in anchors:
                    links.append(a["href"])
                    article_count += 1
                    # check if we have reached the desired number of articles
                    if max_articles and article_count >= max_articles:
                        break

                # check if we have reached the desired number of articles
                if max_articles and article_count >= max_articles:
                    break

                # check if there are more pages to fetch
                next_page = soup.find("a", attrs={"id": "pnnext"})
                if next_page:
                    next_link = Google.ROOT + next_page.get("href")
                    req = Request(next_link, headers=Google.HEADERS)
                    page = urlopen(req).read()
                    soup = BeautifulSoup(page, "html.parser")
                else:
                    break

                time.sleep(1.5)

        fetched_links = pd.DataFrame({"Browser": "Google", "Link": links})
        return fetched_links


class Bing:
    """
    A Class aimed at extracting relevant news article links from Bing.

    Parameters
    ----------
    start_date : {}, default=None
        Specify a start date to be included in the Bing search

    end_date : {}, default=None
        Specify a end date to be included in the Bing search

    duration : {}, default=None
        Specify a specific number of months back from today in the Bing search

    company : {}, default=None
        Specify a start date to be included in the Bing search

    Examples
    --------
    >>> from link_extractor import Bing
    >>> search = Bing(company="Tesla")
    """

    ROOT = "https://www.bing.com/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }

    def __init__(
        self, start_date=None, end_date=None, duration=None, company=None, country="us"
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.company = company
        self.country = country

    async def get_links(self, max_articles=None) -> pd.DataFrame:
        """
        Extracts links from specified browser class and returns a dataframe object

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of Bing articles to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Browser" and "Link"

        Examples
        --------
        >>> from link_extractor import Bing
        >>> search = Bing(company="Tesla")
        >>> links = await search.get_links(max_articles=50)
        """

        # adding date query parameters
        if self.start_date and self.end_date:
            date_range = f"&qdr=h&first={self.start_date}&last={self.end_date}"
        else:
            date_range = ""

        # define full search url
        search = (
            f"{Bing.ROOT}news/search?q={self.company}&cc={self.country}{date_range}"
        )

        # define loop variables
        links = []
        article_count = 0

        # Initiate playwright session and extract links from infinite scroll page
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(10000)
            await page.goto(search)

            # Get the main tag and initial a tags within it
            main_tag = await page.query_selector("main")
            last_count = 0
            while True:
                # Get all "a" tags with class "title"
                a_tags = await main_tag.query_selector_all("a.title")

                # Scroll each "a" tag into view and print its href attribute
                for tag in a_tags[last_count:]:
                    await tag.scroll_into_view_if_needed()
                    href = await tag.get_attribute("href")
                    links.append(href)
                    article_count += 1

                    if max_articles and article_count >= max_articles:
                        break

                if max_articles and article_count >= max_articles:
                    break

                # If no new "a" tags were loaded, stop scrolling
                if len(a_tags) == last_count:
                    break
                last_count = len(a_tags)

                # Scroll to the bottom of the page to load more "a" tags
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1.5)  # Wait for new tags to load

        fetched_links = pd.DataFrame({"Browser": "Bing", "Link": links})
        return fetched_links


### TODO ###

# - Not sure if Class structure optimal
# - Instead of "max_pages", maybe "max_news"? So limit not pages, but links DONE
# - ATM for testing a bit excessive use of "self." - should simplify
# - Make request session more robust by adding cookies, headers etc. Also, there is a way to make google think
#   we come from google.com (so we are not the whole time making direct requests to super specific urls) - not really solved, but a bit (maybe)
# - Spanish results??? Way around it? DONE
