# Imports
import hashlib
import logging
import random
import re
import time
import os
import pandas as pd

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests import Session
from requests.utils import unquote


# Anti-Scraping Measures
ua = UserAgent(fallback="chrome")
DELAY_RANGE = (0.5, 1.5)  # set a random delay between requests to avoid rate limiting

# Logging Config
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGER = logging.getLogger("link_extractor")
LOGGER.setLevel(logging.DEBUG)

link_extractor_handler = logging.FileHandler(os.path.join(LOG_DIR, "link_extractor.log"))
link_extractor_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

LOGGER.addHandler(link_extractor_handler)


class SearchEngines:
    """
    A class aimed to provide a common interface for all the Search Engines (Google, Bing, etc.)
    """

    def __init__(self, start_date=None, end_date=None, duration=None, company=None, country="us"):
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.company = company
        self.country = country

    @abstractmethod
    def get_links(self, max_articles=None, threads=1):
        """
        An abstract method that gets links from the search engine.

        This method should be implemented in child classes to provide search-specific functionality.

        Parameters
        ----------
        max_articles : int, default=None
            Specify the maximum number of articles to scrape. By default, scrapes all available.

        Returns
        -------
        fetched_links : Pandas DataFrame with columns "Search Engine" and "Link"
        """
        pass


class Google(SearchEngines):
    """
    A Class aimed at extracting relevant news article links from Google.

    Examples
    --------
    >>> from link_extractor import Google
    >>> google = Google(company="Tesla")
    """

    ROOT = "https://www.google.com/"
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.44"

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
        query_params = {
            "q": self.company,
            "hl": "en",
            "tbm": "nws",
            "gl": self.country
        }

        # adding date query parameters
        if self.start_date and self.end_date:
            query_params["tbs"] = f"cdr:1,cd_min:{self.start_date},cd_max:{self.end_date}"

        # define full search url
        query_string = "&".join([f"{key}={value}" for key, value in query_params.items()])
        search_url = f"{Google.ROOT}search?{query_string}"

        # create session and extract links from different pages
        with Session() as session:

            # define loop variables
            results = []
            article_count = 0

            while True:
                LOGGER.debug(f"Read articles: {article_count}")

                # create request
                req = Request(search_url, headers={"User-Agent": Google.USER_AGENT})
                LOGGER.debug(f"Starting new HTTPS connection: {search_url}")

                # open the request
                page = urlopen(req)
                LOGGER.info(f"{req.get_method()} {search_url} {page.status}")

                # create beautiful soup object
                soup = BeautifulSoup(page.read(), "lxml")

                # extract HTML anchors in the page
                anchors = soup.find_all("a", {"class": "WlydOe"})

                # loop over all the found links and append
                for a in anchors:

                    # extract information
                    result = {
                        "Search Engine": "Google",
                        "Link": a["href"],
                        "Title": a.find("div", {"role": "heading"}).text,
                        "Source": a.find("span").text,
                    }

                    # add new information to resutls
                    results.append(result)
                    LOGGER.info(result)

                    # increase number of articles read by 1
                    article_count += 1

                    # check if we have reached the desired number of articles
                    if max_articles and article_count >= max_articles:
                        break

                # check if we have reached the desired number of articles
                if max_articles and article_count >= max_articles:
                    break

                # check if there are more pages to fetch
                next_page = soup.find("a", attrs={"id": "pnnext"})
                if not next_page:
                    break

                # update url and continue to next page 
                search_url = Google.ROOT + next_page["href"]
                time.sleep(random.uniform(*DELAY_RANGE))

        return pd.DataFrame(results)


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
        results = []

        article_count = 0
        num_results = 0

        # scrape links from Bing news
        prev_hash = None

        while True:
            # define full search url
            search = f"{Bing.ROOT}news/infinitescrollajax?cc={self.country}&InfiniteScroll=1&q={self.company}&first={num_results}{date_range}"  # specify lang parameter still todo

            with Session() as session:
                req = Request(search, headers={"User-Agent": ua.random})

                LOGGER.debug(f"Starting new HTTPS connection (1): {req.host}") # Might not be needed to have multiple different sessions

                page = urlopen(req)

                LOGGER.info(f"{req.get_method()} {search} {page.status}")

                soup = BeautifulSoup(page.read(), "lxml")

            # compute the hash of the current soup
            soup_hash = hashlib.md5(soup.encode()).hexdigest()

            # compare the hash of the current soup with the previous hash
            if prev_hash and prev_hash == soup_hash:
                break

            prev_hash = soup_hash

            # Extract all the URLs of the news articles from the HTML response
            for article in soup.find_all("div", "news-card"):

                link = article["data-url"]
                title = article["data-title"]
                source = article["data-author"]

                result = {
                        "Search Engine": "Bing",
                        "Link": link,
                        "Title": title,
                        "Source": source,
                    }

                results.append(result)

                LOGGER.info(result)

                article_count += 1

                if max_articles and article_count >= max_articles:
                    break

            if max_articles and article_count >= max_articles:
                break

            if num_results == 200: # This not optimal, have to change
                break

            num_results += 10
            time.sleep(random.uniform(*DELAY_RANGE))

        fetched_links = pd.DataFrame(results)
        return fetched_links


class Yahoo(SearchEngines):
    """
    A Class aimed at extracting relevant news article links from Yahoo.

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
        results = []
        article_count = 0

        # create session and extract links from different pages
        with Session() as session:
            req = Request(search, headers={"User-Agent": ua.random})

            LOGGER.debug(f"Starting new HTTPS connection (1): {req.host}")
            
            page = urlopen(req)

            LOGGER.info(f"{req.get_method()} {search} {page.status}")

            soup = BeautifulSoup(page.read(), "lxml")

            while True:
                # extract HTML anchors in the page
                anchors = soup.find_all("div", "NewsArticle")

                # append each of the "href" to the links
                for a in anchors:
                    # First have to clean the links (yahoo provides them somewhat encoded)
                    messy_link = a.find("a")["href"]
                    unquoted_link = unquote(messy_link)

                    link = re.search(re.compile(r"RU=(.+)\/RK"), unquoted_link).group(1)
                    title = a.find("h4", "s-title").text
                    source = a.find("span", "s-source").text

                    result = {
                        "Search Engine": "Yahoo",
                        "Link": link,
                        "Title": title,
                        "Source": source,
                    }

                    results.append(result)

                    LOGGER.info(result)

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
                    page = urlopen(req)

                    LOGGER.info(f"{req.get_method()} {search} {page.status}")

                    soup = BeautifulSoup(page.read(), "lxml")
                else:
                    break

                time.sleep(random.uniform(*DELAY_RANGE))

        fetched_links = pd.DataFrame(results)
        return fetched_links


def get_all_links(start_date=None, end_date=None, duration=None, company=None, country="us", max_articles=None, threads=1):
    """
    Retrieves links from multiple search engines in parallel and concatenates the results into a single pandas DataFrame.

    Parameters:
    -----------
    start_date : str, optional
        The start date for the search. Defaults to None.
    
    end_date : str, optional
        The end date for the search. Defaults to None.
    
    duration : str, optional
        The duration for the search. Defaults to None.
    
    company : str, optional
        The name of the company to search for. Defaults to None.
    
    country : str, optional
        The country to search in. Defaults to "us".
    
    max_articles : int, optional
        The maximum number of articles to retrieve from each search engine. Defaults to None (no limit).
    
    threads : int, optional
        The number of threads to use for parallelization. Defaults to 1.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the links retrieved from all search engines.

    """
    engines = [Google, Bing, Yahoo]
    args = (start_date, end_date, duration, company, country)
    engine_results = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submitting the tasks to the executor
        futures = [executor.submit(engine(*args).get_links, max_articles) for engine in engines]
        engine_results = [future.result() for future in as_completed(futures)]

    return pd.concat(engine_results, ignore_index=True)


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
