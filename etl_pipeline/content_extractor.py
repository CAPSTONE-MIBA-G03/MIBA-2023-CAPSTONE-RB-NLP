import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import psutil
from fake_useragent import UserAgent
from newspaper import Article, Config
from tqdm import tqdm

# Anti-Scraping Measures
UA = UserAgent(fallback="chrome")
DELAY_RANGE = (0.5, 1.5)  # set a random delay between requests to avoid rate limiting

# Logging Config
LOG_DIR = "logs"

LOGGER = logging.getLogger("content_extractor")
LOGGER.setLevel(logging.DEBUG)

content_extractor_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "content_extractor.log")
)
content_extractor_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

LOGGER.addHandler(content_extractor_handler)

# Newspaper3k Config
config = Config()
config.browser_user_agent = UA.random #"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
config.memoize_articles = False
config.request_timeout = 10


def process_article(article):
    """
    Internal function that is used to parse newspaper3k "Article" Objects
    """

    LOGGER.debug(f"Starting new HTTPS connection (1): {article.source_url}")
    article.download()
    if article.download_exception_msg:
        raise Exception(article.download_exception_msg)
    else:
        LOGGER.info(f"GET {article.source_url} {article.download_state}00")
    article.parse()
    result = {
        "Link": article.url,
        "Title": article.title,
        "Body": article.text,
        "Author": article.authors,
        "Published": article.publish_date,
    }
    time.sleep(random.uniform(*DELAY_RANGE))
    LOGGER.info(result)
    return result


def get_content(links: list):
    """
    Retrieves article content from multiple URLs in parallel and returns a pandas DataFrame with the parsed information.

    Parameters:
    -----------
    links : list
        A list of URLs to retrieve article content from.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing parsed information for each article, including link, title, body, author, and publish date.

    Examples
    --------
    >>> from etl_pipeline.content_extractor import get_content
    >>> from etl_pipeline.link_extractor import Bing, Yahoo, get_all_links
    >>> links = get_all_links()
    >>> results = get_content(links)
    """


    if not isinstance(links, list):
        raise TypeError("Input must be a list.")

    articles = [Article(url=url, config=config) for url in links]
    num_articles = len(articles)

    # Determine optimal number of threads
    cpu_count = psutil.cpu_count()
    mem_avail = psutil.virtual_memory().available
    mem_per_article = 100 * (2**20)  ## 100 Mebibytes (MiB) per article
    max_threads = min(cpu_count, int(mem_avail / mem_per_article), num_articles)
    num_threads = max(1, max_threads)  # Ensure at least one thread
    LOGGER.info(f"Using {num_threads} threads to parse {num_articles} articles")

    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_article, article) for article in articles]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Getting news article info"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                LOGGER.error(f"Download failed because of {e}")
                continue

    article_data = pd.DataFrame(results)
    return article_data


# TODO
# Make this a method of the Search Engine classes? Might make sense
# For some articles, the newspaper3k lib seems to fail on grand scale. Example is MSN!
