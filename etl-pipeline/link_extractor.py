# Imports
import hashlib
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests import Session
from requests.utils import unquote
from tqdm import tqdm

ua = UserAgent(fallback="chrome")
delay_range = (0.5, 1.5)  # set a random delay between requests to avoid rate limiting


class SearchEngines:
    """
    A class aimed at extracting relevant news article links from multiple search engines.
    """

    def __init__(
        self, start_date=None, end_date=None, duration=None, company=None, country="us"
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.company = company
        self.country = country

    def get_links(self, max_articles=None, threads=1):
        """
        Extracts links from specified search engine class and returns a dataframe object

        Parameters
        ----------
        max_articles : int, default=None
            Specify the maximum number of articles to scrape. By default, scrapes all available.

        threads : int, default=1
            The number of threads to use for scraping. By default, no multithreading is performed.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Search Engine" and "Link"
        """
        engines = [Google, Bing, Yahoo]
        engine_results = []

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submitting the tasks to the executor
            futures = [
                executor.submit(
                    engine(
                        start_date=self.start_date,
                        end_date=self.end_date,
                        duration=self.duration,
                        company=self.company,
                        country=self.country,
                    ).get_links,
                    max_articles,
                )
                for engine in engines
            ]

            for future in as_completed(futures):
                # Collecting the results
                engine_results.append(future.result())

        fetched_links = pd.concat(engine_results, ignore_index=True)
        return fetched_links


class Google(SearchEngines):
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
    >>> google = Google(company="Tesla")
    """

    ROOT = "https://www.google.com/"
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_links(self, max_articles=None) -> pd.DataFrame:
        """
        Extracts links from specified search engine class and returns a dataframe object

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of google articles to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Search Engine" and "Link"

        Examples
        --------
        >>> from link_extractor import Google
        >>> google = Google(company="Tesla")
        >>> links = google.get_links(max_articles=50)
        """

        # adding date query parameters
        if self.start_date and self.end_date:
            date_range = f"&tbs=cdr:1,cd_min:{self.start_date},cd_max:{self.end_date}"
        else:
            date_range = ""

        # define full search url
        search = f"{Google.ROOT}search?q={self.company}&hl=en&tbm=nws&gl={self.country}{date_range}"

        # define loop variables
        links = []
        titles = []
        sources = []
        article_count = 0

        # create session and extract links from different pages
        with Session() as session:
            req = Request(search, headers={"User-Agent": Google.USER_AGENT})
            page = urlopen(req).read()
            soup = BeautifulSoup(page, "lxml")

            while True:
                # extract HTML anchors in the page
                # search = soup.find("div", {"id": "search"})
                anchors = soup.find_all("a", {"class": "WlydOe"})

                # append each of the "href" to the links
                for a in anchors:
                    links.append(a["href"])
                    titles.append(a.find("div", {"role": "heading"}).text)
                    sources.append(a.find("span").text)  # Fix (not getting it always)

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
                    next_link = Google.ROOT + next_page["href"]
                    req = Request(next_link, headers={"User-Agent": Google.USER_AGENT})
                    page = urlopen(req).read()
                    soup = BeautifulSoup(page, "lxml")
                else:
                    break

                time.sleep(random.uniform(*delay_range))

        fetched_links = pd.DataFrame(
            {
                "Search Engine": "Google",
                "Link": links,
                "Title": titles,
                "Source": sources,
            }
        )
        return fetched_links


class Bing(SearchEngines):
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
    >>> bing = Bing(company="Tesla")
    """

    ROOT = "https://www.bing.com/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_links(self, max_articles=None) -> pd.DataFrame:
        """
        Extracts links from specified search engine class and returns a dataframe object

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of Bing articles to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Search Engine" and "Link"

        Examples
        --------
        >>> from link_extractor import Bing
        >>> bing = Bing(company="Tesla")
        >>> links = bing.get_links(max_articles=50)
        """

        # adding date query parameters
        if self.start_date and self.end_date:
            date_range = f"&qdr=h&first={self.start_date}&last={self.end_date}"
        else:
            date_range = ""

        # define loop variables
        links = []
        titles = []
        sources = []

        article_count = 0
        num_results = 0

        # scrape links from Bing news
        prev_hash = None

        while True:
            # define full search url
            search = f"{Bing.ROOT}news/infinitescrollajax?cc={self.country}&InfiniteScroll=1&q={self.company}&first={num_results}{date_range}"  # specify lang parameter still todo

            with Session() as session:
                req = Request(search, headers={"User-Agent": ua.random})
                page = urlopen(req).read()
                soup = BeautifulSoup(page, "lxml")

            # compute the hash of the current soup
            soup_hash = hashlib.md5(soup.encode()).hexdigest()

            # compare the hash of the current soup with the previous hash
            if prev_hash and prev_hash == soup_hash:
                break

            prev_hash = soup_hash

            # Extract all the URLs of the news articles from the HTML response
            for article in soup.find_all("div", "news-card"):
                links.append(article["data-url"])
                titles.append(article["data-title"])
                sources.append(article["data-author"])

                article_count += 1

                if max_articles and article_count >= max_articles:
                    break

            if max_articles and article_count >= max_articles:
                break

            if num_results == 200:
                break

            num_results += 10
            time.sleep(random.uniform(*delay_range))

        fetched_links = pd.DataFrame(
            {
                "Search Engine": "Bing",
                "Link": links,
                "Title": titles,
                "Source": sources,
            }
        )
        return fetched_links


class Yahoo(SearchEngines):
    """
    A Class aimed at extracting relevant news article links from Yahoo.

    Parameters
    ----------
    start_date : {}, default=None
        Specify a start date to be included in the Yahoo search

    end_date : {}, default=None
        Specify a end date to be included in the Yahoo search

    duration : {}, default=None
        Specify a specific number of months back from today in the Yahoo search

    company : {}, default=None
        Specify a start date to be included in the Yahoo search

    Examples
    --------
    >>> from link_extractor import Yahoo
    >>> yahoo = Yahoo(company="Tesla")
    """

    ROOT = "https://news.search.yahoo.com/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_links(self, max_articles=None) -> pd.DataFrame:
        """
        Extracts links from specified search engine class and returns a dataframe object

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of Yahoo articles to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Search Engine" and "Link"

        Examples
        --------
        >>> from link_extractor import Yahoo
        >>> yahoo = Yahoo(company="Tesla")
        >>> links = yahoo.get_links(max_articles=50)
        """

        # adding date query parameters
        if self.start_date and self.end_date:
            date_range = f"&tbs=cdr:1,cd_min:{self.start_date},cd_max:{self.end_date}"  # not updated yet!
        else:
            date_range = ""

        # define full search url
        search = f"{Yahoo.ROOT}search?p={self.company}&fr=news&country={self.country}{date_range}&lang=en-US"

        # define loop variables
        links = []
        titles = []
        sources = []
        article_count = 0

        # create session and extract links from different pages
        with Session() as session:
            req = Request(search, headers={"User-Agent": ua.random})
            page = urlopen(req).read()
            soup = BeautifulSoup(page, "lxml")

            while True:
                # extract HTML anchors in the page
                anchors = soup.find_all("div", "NewsArticle")

                # append each of the "href" to the links
                for a in anchors:
                    # First have to clean the links (yahoo provides them somewhat encoded)
                    link = a.find("a")["href"]
                    unquoted_link = unquote(link)
                    cleaned_link = re.search(re.compile(r"RU=(.+)\/RK"), unquoted_link).group(1)
                    links.append(cleaned_link)

                    try:
                        title = a.find("h4", "s-title").text
                        titles.append(title)
                    except:
                        titles.append("None")

                    try:
                        source = a.find("span", "s-source").text
                        sources.append(source)
                    except:
                        sources.append("None")

                    article_count += 1

                    # check if we have reached the desired number of articles
                    if max_articles and article_count >= max_articles:
                        break

                # check if we have reached the desired number of articles
                if max_articles and article_count >= max_articles:
                    break

                # check if there are more pages to fetch
                next_page = soup.find("a", attrs={"class": "next"})
                if next_page:
                    next_link = next_page["href"]
                    req = Request(next_link, headers={"User-Agent": ua.random})
                    page = urlopen(req).read()
                    soup = BeautifulSoup(page, "lxml")
                else:
                    break

                time.sleep(random.uniform(*delay_range))

        fetched_links = pd.DataFrame(
            {
                "Search Engine": "Yahoo",
                "Link": links,
                "Title": titles,
                "Source": sources,
            }
        )
        return fetched_links


### TODO ###

# - Not sure if Class structure optimal
# - Instead of "max_pages", maybe "max_news"? So limit not pages, but links DONE
# - ATM for testing a bit excessive use of "self." - should simplify
# - Make request session more robust by adding cookies, headers etc. Also, there is a way to make google think
#   we come from google.com (so we are not the whole time making direct requests to super specific urls) - not really solved, but a bit (maybe)
# - Spanish results??? Way around it? DONE
# Google only takes actual actual UA. Fake ones not working. Work in Bing tho.
# WRITE TESTS / ADD LOGGING!!!
# Try to filter by dates when getting the links themselves