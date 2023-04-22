import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import psutil
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from newspaper import Article, Config
from tqdm import tqdm
from user_agent import generate_user_agent

# Anti-Scraping Measures
UA = UserAgent(fallback="chrome")
DELAY_RANGE = (0.5, 1.5)  # set a random delay between requests to avoid rate limiting

# Logging Config
LOG_DIR = "logs"

LOGGER = logging.getLogger("content_extractor")
LOGGER.setLevel(logging.DEBUG)

content_extractor_handler = logging.FileHandler(os.path.join(LOG_DIR, "content_extractor.log"))
content_extractor_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

LOGGER.addHandler(content_extractor_handler)

TIMEOUT = 10

# Newspaper3k Config
config = Config()
config.browser_user_agent = (UA.random)  # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
config.memoize_articles = False
config.request_timeout = TIMEOUT


def _process_with_bs(url):
    # Inspired by: https://medium.com/analytics-vidhya/web-scraping-news-data-rss-feeds-python-and-google-cloud-platform-7a0df2bafe44

    try:
        #request the article url to get the web page content.
        article = requests.get(url, headers={"User-Agent": generate_user_agent()}, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        err = "Request to {url} timed out after {TIMEOUT} seconds"
        LOGGER.error(err)
        return {"url": url, "title": err, "body": ""}
    
    # create BeautifulSoup object
    articles = BeautifulSoup(article.content, "lxml")
    articles_body = articles.findAll("body")

    # 0. extract title
    title = articles_body[0].find("h1").text

    # 1. extract all paragraph elements inside the page body
    p_blocks = articles_body[0].findAll("p")

    # 2. for each paragraph, construct its patents elements hierarchy
    # Create a dataframe to collect p_blocks data
    p_blocks_df = pd.DataFrame()
    for i in range(0, len(p_blocks)):
        # 2.1 Loop trough paragraph parents to extract its element name and id
        parents_list = []

        for parent in p_blocks[i].parents:
            # Extract the parent id attribute if it exists
            parent_id = parent.get("id", "")

            # Append the parent name and id to the parents table
            parents_list.append(parent.name + "id: " + parent_id)

        # 2.2 Construct parents hierarchy
        parent_element_list = ["" if (x == "None" or x is None) else x for x in parents_list]
        # parent_element_list.reverse()  # uncomment if need to debug
        parent_hierarchy = " -> ".join(parent_element_list)

        #   Append data table with the current paragraph data
        d = pd.DataFrame(
            [
                {
                    "element_name": p_blocks[i].name,
                    "parent_hierarchy": parent_hierarchy,
                    "element_text": p_blocks[i].text,
                    "element_text_count": len(str(p_blocks[i].text)),
                }
            ]
        )

        p_blocks_df = pd.concat([d, p_blocks_df], ignore_index=True, sort=False)

    # when no p's are found, then return empty body
    if len(p_blocks_df) <= 0:
        return {"url": url, "title": title, "body": ""}

    # 3. concatenate paragraphs under the same parent hierarchy
    df_groupby_parent_hierarchy = p_blocks_df.groupby(by=["parent_hierarchy"])
    df_groupby_parent_hierarchy_sum = df_groupby_parent_hierarchy[["element_text_count"]].sum()
    df_groupby_parent_hierarchy_sum.reset_index(inplace=True)

    # 4. count paragraphs length
    maxid = df_groupby_parent_hierarchy_sum.loc[df_groupby_parent_hierarchy_sum["element_text_count"].idxmax(),"parent_hierarchy",]

    # 5. select the longest paragraph as the main article
    body_list = p_blocks_df.loc[p_blocks_df["parent_hierarchy"] == maxid, "element_text"].to_list()
    body_list.reverse()
    body = "\n".join(body_list)

    return {"url": url, "title": title, "body": body}


def _process_with_n3k(article):
    article.download()
    if article.download_exception_msg:
        raise Exception(article.download_exception_msg)

    LOGGER.info(f"GET {article.source_url} {article.download_state}00")
    article.parse()


def _process_article(url, n3k_article):
    LOGGER.debug(f"Starting new HTTPS connection (1): {n3k_article.source_url}")

    # Process with n3k library
    LOGGER.debug("Processing using newspaper3k library")
    _process_with_n3k(n3k_article)
    result = {
        "n3k_link": n3k_article.url,
        "n3k_title": n3k_article.title,
        "n3k_body": n3k_article.text,
        "n3k_author": n3k_article.authors,
        "n3k_published": n3k_article.publish_date,
    }

    # Process with Beautiful Soup
    LOGGER.debug("Processing using Beautiful Soup")
    bs_article = _process_with_bs(url)
    result.update(
        {
            "bs_link": bs_article["url"],
            "bs_title": bs_article["title"],
            "bs_body": bs_article["body"],
        }
    )

    # time.sleep(random.uniform(*DELAY_RANGE))
    LOGGER.info(result)

    return result


def get_content(links: list) -> dict:
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

    articles = [(url, Article(url=url, config=config)) for url in links]
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
        futures = [executor.submit(_process_article, url, article) for url, article in articles]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Getting news article info"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                LOGGER.error(f"Download failed because of {e}")
                continue

    return results


# TODO
# Make this a method of the Search Engine classes? Might make sense
# For some articles, the newspaper3k lib seems to fail on grand scale. Example is MSN!
