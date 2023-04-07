import logging
import random
import time

import pandas as pd
from fake_useragent import UserAgent
from newspaper import Article, Config
from tqdm import tqdm

ua = UserAgent(fallback="chrome")
delay_range = (1, 3)  # set a random delay between requests to avoid rate limiting

# Logging Config
content_extractor_logger = logging.getLogger("content_extractor2")
content_extractor_logger.setLevel(logging.DEBUG)

content_extractor_handler = logging.FileHandler("logs/content_extractor2.log")
content_extractor_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
content_extractor_logger.addHandler(content_extractor_handler)


def get_content(links: list):
    if not isinstance(links, list):
        raise TypeError("Input must be a list.")

    results = []

    for link in tqdm(links, desc="Getting news article info"):

        try:
            config = Config()
            config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
            config.memoize_articles = False
            config.request_timeout = 10

            article = Article(link, config=config)

            content_extractor_logger.debug(f"Starting new HTTPS connection (1): {article.source_url}")

            article.download()
            if article.download_exception_msg:
                raise Exception(article.download_exception_msg)

            else:
                content_extractor_logger.info(f"GET {article.source_url} {article.download_state}00")

            article.parse()

            # Put Logger to see if request was sucessfull

            result = {
                "Link": article.url,
                "Title": article.title,
                "Body": article.text,
                "Author": article.authors,
                "Published": article.publish_date
            }

            results.append(result)

            content_extractor_logger.info(result)

            time.sleep(random.randint(*delay_range))

        except Exception as e:
            content_extractor_logger.error(f"Download failed on URL {link} because of {e}")
            continue

    article_data = pd.DataFrame(results)
    return article_data

# TODO
# Make this a method of the Search Engine classes? Might make sense
# For some articles, the newspaper3k lib seems to fail on grand scale. Example is MSN!