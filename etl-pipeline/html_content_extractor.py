import random
import time

import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from newspaper import Article
from playwright.sync_api import sync_playwright

ua = UserAgent()
delay_range = (1, 5)  # set a random delay between requests to avoid rate limiting


def get_content(links: list):
    if not isinstance(links, list):
        raise TypeError("Input must be a list.")

    url = []
    title = []
    text = []
    author = []
    date = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        browser_context = browser.new_context(user_agent=ua.random)
        page = browser_context.new_page()

        for link in links:
            page.goto(link, wait_until="networkidle")

            # try to accept full marketing consent
            consent_accept_selectors = {
                "onetrust-cookiepro": "#onetrust-accept-btn-handler",
                "onetrust-enterprise": "#accept-recommended-btn-handler",
                "cookiebot": "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
                "cookiehub": "button.ch2-allow-all-btn",
                "typo3-wacon": ".waconcookiemanagement .cookie-accept",
                "cookiefirst": "[data-cookiefirst-action='accept']",
                "osano": ".osano-cm-accept-all",
                "orejime": ".orejime-Button--save",
                "axeptio": "#axeptio_btn_acceptAll",
                "civic-uk-cookie-control": "#ccc-notify-accept",
                "usercentrics": "[data-testid='uc-accept-all-button']",
                "cookie-yes": "[data-cky-tag='accept-button']",
                "secure-privacy": ".evSpAcceptBtn",
                "quantcast": "#qc-cmp2-ui button[mode='primary']",
                "didomi": "#didomi-notice-agree-button",
                "trustarc-truste": "#truste-consent-button",
                "non-specific-custom": "#AcceptCookiesButton, #acceptCookies, .cookie-accept, #cookie-accept, .gdpr-cookie--accept-all, button[class*='accept'], button[id*='accept'], [class*='accept'], [id*='accept'], #cookiebanner button, [class*='cookie']",
            }

            consent_manager = "none detected"  # default value
            for k in consent_accept_selectors.keys():
                if page.locator(consent_accept_selectors[k]).count() > 0:
                    # if the count is > 0 we've found our button to click
                    consent_manager = k
                    # we try to click and explicitly wait for navigation as some pages will reload after accepting cookies
                    try:
                        with page.expect_navigation(
                            wait_until="networkidle", timeout=15000
                        ):
                            page.click(consent_accept_selectors[k], delay=10)
                    except Exception as e:
                        print(url, e)
                    break

            time.sleep(1)

            html = page.content()

            # use newspaper to parse the HTML
            article = Article(link)
            article.set_html(html)
            article.parse()

            url.append(article.url)
            title.append(article.title)
            text.append(article.text)
            author.append(article.authors)
            date.append(article.publish_date)

            authors = article.authors

            # do something with the data

            print(title)
            print(text)
            print(authors)

            time.sleep(random.randint(*delay_range))


get_content(
    [
        "https://www.bbc.com/news/business-64867287",
        "https://www.morgenpost.de/wirtschaft/article238059211/Tesla-trifft-mit-im-ersten-Quartal-in-etwa-die-Erwartungen.html",
    ]
)
