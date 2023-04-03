import random
import time

import pandas as pd
from fake_useragent import UserAgent
from newspaper import Article, Config
from tqdm import tqdm

ua = UserAgent(fallback="chrome")
delay_range = (1, 3)  # set a random delay between requests to avoid rate limiting


def get_content(links: list):
    if not isinstance(links, list):
        raise TypeError("Input must be a list.")

    url = []
    title = []
    text = []
    author = []
    date = []
    blocked_or_error = []

    for link in tqdm(links, desc="Getting news article info"):
        
        try:
            config = Config()
            config.browser_user_agent = ua.random
            config.request_timeout = 10

            article = Article(link, config=config)
            article.download()
            article.parse()

            url.append(article.url)
            title.append(article.title)
            text.append(article.text)
            author.append(article.authors)
            date.append(article.publish_date)

            time.sleep(random.randint(*delay_range))

        except:

            blocked_or_error.append(article.url)
            continue

    article_data = pd.DataFrame(
        {
            "Link": url,
            "Title": title,
            "Body": text,
            "Author": author,
            "Published": date
        }
    )

    if len(blocked_or_error) > 1:

        print("The following news sources could not be accessed or resulted in an error:")
        for error in blocked_or_error:
            print(error, "\n")

    return article_data
