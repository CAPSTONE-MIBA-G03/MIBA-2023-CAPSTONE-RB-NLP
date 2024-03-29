# Imports
import hashlib
import logging
import os
import random
import re
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus  # unquote from urllib breaks code
from urllib.request import Request, urlopen

import arrow
import psutil
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests import Session
from requests.utils import unquote  # quote from requests breaks code

# Anti-Scraping Measures
UA = UserAgent(fallback="chrome")
DELAY_RANGE = (1.0, 2.5)  # set a random delay between requests to avoid rate limiting

# Logging Config
LOG_DIR = "logs"

LOGGER = logging.getLogger("link_extractor")
LOGGER.setLevel(logging.DEBUG)

link_extractor_handler = logging.FileHandler(os.path.join(LOG_DIR, "link_extractor.log"))
link_extractor_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

LOGGER.addHandler(link_extractor_handler)


class SearchEngines:
    """
    A class aimed to provide a common interface for all the Search Engines (Google, Bing, etc.)
    """

    def __init__(self, query=None, start_date=None, end_date=None, duration=None, country="us"):
        self.query = query
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.country = country

    @abstractmethod
    def get_links(self, max_articles=None) -> list:
        """
        An abstract method that gets links from the search engine.

        This method should be implemented in child classes to provide search-specific functionality.

        Parameters
        ----------
        max_articles : int, default=None
            Specify the maximum number of articles to scrape. By default, scrapes all available.

        Returns
        -------
        results : list
            List of dictionaries with keys "engine", "se_link", "se_title", and "se_source".
        """

        pass


class Google(SearchEngines):
    """
    A Class aimed at extracting relevant news article links from Google.

    Parameters
    ----------
    query : str, default=None
        Specify a start date to be included in the Google search.

    start_date : {}, default=None
        Specify a start date to be included in the Google search. Format: YYYYMMDD

    end_date : {}, default=None
        Specify a end date to be included in the Google search. Format: YYYYMMDD

    duration : int, default=None
        Specify a specific number of months back from today in the Google search.

    country : str, default="us"
        Specify a country to be included in the Google search. Defaults to "us".

    Methods
    -------
    get_links(max_articles=None)
        Extracts links from specified search engine class.

    Examples
    --------
    >>> from etl_pipeline.link_extractor import Google
    >>> google = Google(query="Tesla")

    References
    -----
    Inspired by and adapted from https://www.youtube.com/watch?v=Ikf6Xdox0Go

    IMPORTANT!
    -----

    The user agent for Google might have to be hardcoded as of now depending on the OS you are using.
    Scraping Google is challenging. Without a user agent, Google will block your request. Using a fake user agent has also not been successful for now.
    To find your user agent, google "What is my user agent" and copy the string that is returned.
    """

    ROOT = "https://www.google.com/"
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.44"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_links(self, max_articles=None) -> list:
        """
        Extracts links from specified search engine class.

        Parameters
        ----------
        max_articles : int, default=None
            Specify the maximum number of google articles to scrape. By default, scrapes all available.

        Returns
        -------
        results : list
            List of dictionaries with keys "engine", "se_link", "se_title", and "se_source".

        Examples
        --------
        >>> from etl_pipeline.link_extractor import Google
        >>> google = Google(query="Tesla")
        >>> links = google.get_links(max_articles=50)
        """
        query_params = {
            "q": quote_plus(self.query),  # Enables search to have spaces
            "hl": "en",
            "tbm": "nws",
            "gl": self.country,
        }

        if all([self.start_date, self.end_date, self.duration]):
            raise SyntaxError("Duration can't be an input if a start and/or end date is also specified")

        # adding date query parameters
        if self.start_date and self.end_date:
            query_params["tbs"] = f"cdr:1,cd_min:{self.start_date},cd_max:{self.end_date}"

        elif self.duration:
            today = arrow.now("Europe/Paris")
            delta = today.shift(months=-self.duration)
            query_params["tbs"] = f"cdr:1,cd_min:{delta.format('YYYYMMDD')},cd_max:{today.format('YYYYMMDD')}"

        # define full search url
        query_string = "&".join([f"{key}={value}" for key, value in query_params.items()])
        search_url = f"{Google.ROOT}search?{query_string}"
        LOGGER.info(f"Search URL: {search_url}")

        # define loop variables
        results = []
        article_count = 0

        # create session and extract links from different pages
        with Session() as session:
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
                    link = a.get("href")
                    title = a.find("div", {"role": "heading"})
                    description = a.find("div", {"class": "GI74Re"})
                    source = a.find("span")

                    result = {
                        "engine": "Google",
                        "se_link": link,
                        "se_title": title.text,
                        "se_description": description.text if description else None,
                        "se_source": source.text,
                    }

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
                next_page = soup.find("a", {"id": "pnnext"})
                if not next_page:
                    break

                # update url and continue to next page
                search_url = Google.ROOT + next_page["href"]
                time.sleep(random.uniform(*DELAY_RANGE))

        return results


class Bing(SearchEngines):
    """
    A Class aimed at extracting relevant news article links from Bing.
    Circumvents need for headless browser by making direct requests to Bing's "infinite scroll" search results.

    Parameters
    ----------
    query : {}, default=None
        Specify a start date to be included in the Bing search.

    start_date : {}, default=None
        Specify a start date to be included in the Bing search.

    end_date : {}, default=None
        Specify a end date to be included in the Bing search.

    duration : {}, default=None
        Specify a specific number of months back from today in the Bing search.

    country : {}, default="us"
        Specify a country to be included in the Bing search. Defaults to "us".

    Methods
    -------
    get_links(max_articles=None)
        Extracts links from specified search engine class.

    Examples
    --------
    >>> from etl_pipeline.link_extractor import Bing
    >>> bing = Bing(query="Tesla")
    """

    ROOT = "https://www.bing.com/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_links(self, max_articles=None) -> list:
        """
        Extracts links from specified search engine class.

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of Bing articles to scrape. By default, scrapes all available.

        Returns
        -------
        results : list
            List of dictionaries with keys "engine", "se_link", "se_title", and "se_source".

        Examples
        --------
        >>> from etl_pipeline.link_extractor import Bing
        >>> bing = Bing(query="Tesla")
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
        num_results = 1

        # scrape links from Bing news
        prev_hash = None

        while True:
            if num_results >= 211:
                break
            # define full search url
            search = f"{Bing.ROOT}news/infinitescrollajax?cc={self.country}&InfiniteScroll=1&q={quote_plus(self.query)}&first={num_results}{date_range}"  # specify lang parameter still todo

            with Session() as session:
                req = Request(search, headers={"User-Agent": UA.random})

                LOGGER.debug(
                    f"Starting new HTTPS connection (1): {req.host}"
                )  # Might not be needed to have multiple different sessions

                page = urlopen(req)

                LOGGER.info(f"{req.get_method()} {search} {page.status}")

                soup = BeautifulSoup(page.read(), "lxml")

            # compute the hash of the current soup
            soup_hash = hashlib.md5(soup.encode()).hexdigest()

            # compare the hash of the current soup with the previous hash and break if same
            if prev_hash and prev_hash == soup_hash:
                break

            prev_hash = soup_hash

            # Extract all the URLs of the news articles from the HTML response
            for article in soup.find_all("div", {"class": "news-card"}):
                link = article.get("data-url")
                title = article.get("data-title")
                description = article.find("div", {"class": "snippet"}).get("title")
                source = article.get("data-author")

                result = {
                    "engine": "Bing",
                    "se_link": link,
                    "se_title": title,
                    "se_description": description,
                    "se_source": source,
                }

                results.append(result)
                LOGGER.info(result)

                article_count += 1

                if max_articles and article_count >= max_articles:
                    break

            if max_articles and article_count >= max_articles:
                break

            num_results += 10
            time.sleep(random.uniform(*DELAY_RANGE))

        return results


class Yahoo(SearchEngines):
    """
    A Class aimed at extracting relevant news article links from Yahoo.

    Parameters
    ----------
    query : {}, default=None
        Specify a start date to be included in the Yahoo search.

    start_date : {}, default=None
        Specify a start date to be included in the Yahoo search.

    end_date : {}, default=None
        Specify a end date to be included in the Yahoo search.

    duration : {}, default=None
        Specify a specific number of months back from today in the Yahoo search.

    country : {}, default="us"
        Specify a country to be included in the Yahoo search. Defaults to "us".
    
    Methods
    -------
    get_links(max_articles=None)
        Extracts links from specified search engine class.

    Examples
    --------
    >>> from etl_pipeline.link_extractor import Yahoo
    >>> yahoo = Yahoo(query="Tesla")
    """

    ROOT = "https://news.search.yahoo.com/"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_links(self, max_articles=None) -> list:
        """
        Extracts links from specified search engine class.

        Parameters
        ----------
        max_articles : {}, default=None
            Specify the maximum number of Yahoo articles to scrape. By default, scrapes all available.

        Returns
        -------
        results : list
            List of dictionaries with keys "engine", "se_link", "se_title", and "se_source".

        Examples
        --------
        >>> from etl_pipeline.link_extractor import Yahoo
        >>> yahoo = Yahoo(query="Tesla")
        >>> links = yahoo.get_links(max_articles=50)
        """

        # adding date query parameters
        if self.start_date and self.end_date:
            date_range = f"&tbs=cdr:1,cd_min:{self.start_date},cd_max:{self.end_date}"  # not updated yet!
        else:
            date_range = ""

        # define full search url
        search = f"{Yahoo.ROOT}search?p={quote_plus(self.query)}&fr=news&country={self.country}{date_range}&lang=en-US"

        # define loop variables
        results = []
        article_count = 0

        # create session and extract links from different pages
        with Session() as session:
            req = Request(search, headers={"User-Agent": UA.random})

            LOGGER.debug(f"Starting new HTTPS connection (1): {req.host}")

            page = urlopen(req)

            LOGGER.info(f"{req.get_method()} {search} {page.status}")

            soup = BeautifulSoup(page.read(), "lxml")

            while True:
                # extract HTML anchors in the page
                anchors = soup.find_all("div", {"class": "NewsArticle"})

                for a in anchors:
                    # First have to clean the links (yahoo provides them somewhat encoded)
                    try:
                        messy_link = a.find("a")["href"]
                        unquoted_link = unquote(messy_link)
                        link = re.search(re.compile(r"RU=(.+)\/RK"), unquoted_link).group(1)
                    except:
                        link = None

                    title = a.find("h4", {"class": "s-title"})
                    source = a.find("span", {"class": "s-source"})
                    description = a.find("p", {"class": "s-desc"})

                    result = {
                        "engine": "Yahoo",
                        "se_link": link,
                        "se_title": title.text,
                        "se_description": description.text,
                        "se_source": source.text,
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
                    req = Request(next_link, headers={"User-Agent": UA.random})
                    page = urlopen(req)

                    LOGGER.info(f"{req.get_method()} {search} {page.status}")

                    soup = BeautifulSoup(page.read(), "lxml")
                else:
                    break

                time.sleep(random.uniform(*DELAY_RANGE))

        return results


def get_all_links(
    query=None,
    start_date=None,
    end_date=None,
    duration=None,
    country="us",
    max_articles=None,
    engines=[Google, Bing, Yahoo],
) -> list:
    """
    Wrapper function that calls the "get_links" method on multiple search engine classes in parallel.

    Parameters:
    -----------
    query : str, optional
        The name of the company to search for. Defaults to None.

    start_date : str, optional
        The start date for the search. Defaults to None.

    end_date : str, optional
        The end date for the search. Defaults to None.

    duration : str, optional
        The duration for the search. Defaults to None.

    country : str, optional
        The country to search in. Defaults to "us".

    max_articles : int, optional
        The maximum number of articles to retrieve from each search engine. Defaults to None (no limit).

    engines : list, optional
        A list containing prefered search engines. Defaults to all available.

    Returns:
    --------
    results : list
        List of dictionaries with keys "engine", "se_link", "se_title", and "se_source" for all specified search engines.

    Examples
    --------
    >>> from etl_pipeline.link_extractor import Bing, Yahoo, get_all_links
    >>> bing_yahoo = get_all_links(query="Tesla",  max_articles=20, engines=[Bing, Yahoo])
    """

    # Important! The order of the arguments must match the order of the arguments in the "get_links" method
    args = (query, start_date, end_date, duration, country)
    engine_results = []

    num_engines = len(engines)
    num_threads = min(psutil.cpu_count(), num_engines)

    LOGGER.info(f"Using {num_threads} threads, accessing {num_engines} search engines")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submitting the tasks to the executor
        futures = [executor.submit(engine(*args).get_links, max_articles) for engine in engines]
        engine_results = [future.result() for future in as_completed(futures)]

    all_results = [results for sublist in engine_results for results in sublist]
    unique_results = list({v["se_link"]: v for v in all_results}.values())
    LOGGER.info(f"Found {len(unique_results)} unique articles out of {len(all_results)} total")
    return unique_results
