import numpy as np
import pandas as pd


def clean_content(df):
    """
    Function to clean the content of extracted articles.
    Expects as input the dataframe returned by the content_extractor.py module.

    The following cleaning steps are performed:
    - Drops identical columns and renames relevant columns
    - Replaces all '\n' and '\t', multiple spaces, and leading and trailing spaces with a single space
    - Replaces all entries except bodies which contain unwanted words with empty strings for later removal
    - Replaces entries which are too short as empty string for later removal
    - Drops all rows that have no title, description, body or paragraph
    - Removes all websites, emails, phone numbers, and html tags

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe with the content extracted by the content_extractor.py module.

    Returns
    -------
    df_clean : pandas.DataFrame
    A pandas dataframe with the cleaned content and following columns:
    - link: the link to the article
    - title: the title of the article
    - description: the description of the article
    - body: the body of the article
    - source: the source of the article
    - paragraph: the paragraph of the article
    
    Notes
    -----
    This function should be considered work in progress and might be too harsh or too lenient in its cleaning steps.
    Downstream models will likely benefit from a different (and more sophisticated) cleaning approach.
    """

    df_dirty = df.copy()
    df_dirty.fillna("", inplace=True)

    # Drop identical columns and rename relevant columns
    df_dirty.drop(["n3k_link", "bs_link"], axis=1, inplace=True)
    df_dirty.rename(
        columns={
            "se_link": "link",
            "se_description": "description",
            "bs_paragraph": "paragraph",
            "se_source": "source",
            },
        inplace=True,
    )

    # Regex patterns to detect websites, emails, phone numbers, html tags, and empty strings
    website_pattern = r"(?:http[s]?://)?www\.[^\s.]+\.[^\s]{2,}|^https?:\/\/.*[\r\n]*"
    email_pattern = r"[\w.-]+@[\w.-]+\.[\w.-]+"
    phone_pattern = r"\+?\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"

    indent_pattern = r"\n|\t| +"
    html_pattern = r"<.*?>"

    empty_string_pattern = r"^\s*$"

    replacement_pattern = rf"(?:{indent_pattern}|{html_pattern})"
    removal_pattern = rf"(?:{website_pattern}|{email_pattern}|{phone_pattern}|{html_pattern})"

    # Replace all '\n' and '\t', multiple spaces, and leading and trailing spaces with a single space
    for col in ["n3k_title", "n3k_body", "bs_title", "bs_body", "paragraph", "description"]:
        df_dirty[col] = df_dirty[col].str.replace(replacement_pattern, " ", regex=True).str.strip()

    undesireable_phrases = [
        "javascript", "cookie", "cookies", "explorer", "are you a robot", "subscribe",
        "register", "login", "sign in", "sign up", "log in", "sign out", "log out", "privacy",
        "terms", "contact", "about", "help", "feedback", "careers", "advertise", "rate us", 
        "subscribe to unlock", "give us feedback", "free download", "all rights reserved", "©",
        "about us", "contact us", "privacy policy",
        ]
    
    # Flag entries containing any of the unwanted words as NAN for later removal
    df_dirty.loc[df_dirty["n3k_title"].str.contains("|".join(undesireable_phrases), case=False), "n3k_title"] = np.nan
    df_dirty.loc[df_dirty["bs_title"].str.contains("|".join(undesireable_phrases), case=False), "bs_title"] = np.nan
    df_dirty.loc[df_dirty["se_title"].str.contains("|".join(undesireable_phrases), case=False), "se_title"] = np.nan
    df_dirty.loc[df_dirty["paragraph"].str.contains("|".join(undesireable_phrases), case=False), "paragraph"] = np.nan
    df_dirty.loc[df_dirty["description"].str.contains("|".join(undesireable_phrases), case=False), "description"] = np.nan

    # Flag entries which are too short as NAN for later removal
    df_dirty.loc[df_dirty["n3k_title"].str.len() < 20, "n3k_title"] = np.nan
    df_dirty.loc[df_dirty["bs_title"].str.len() < 20, "bs_title"] = np.nan
    df_dirty.loc[df_dirty["se_title"].str.len() < 20, "se_title"] = np.nan
    df_dirty.loc[df_dirty["description"].str.len() < 100, "description"] = np.nan
    df_dirty.loc[df_dirty["bs_body"].str.len() < 400, "bs_body"] = np.nan
    df_dirty.loc[df_dirty["n3k_body"].str.len() < 400, "n3k_body"] = np.nan
    df_dirty.loc[df_dirty["paragraph"].str.len() < 150, "paragraph"] = np.nan

    df_clean = df_dirty.copy()

    df_clean["n3k_body_len"] = df_clean["n3k_body"].str.len()
    df_clean["bs_body_len"] = df_clean["bs_body"].str.len()

    # Creating a column with the length of the n3k_title, bs_title and se_title
    df_clean["n3k_title_len"] = df_clean["n3k_title"].str.len()
    df_clean["bs_title_len"] = df_clean["bs_title"].str.len()
    df_clean["se_title_len"] = df_clean["se_title"].str.len()

    df_clean["title"] = ""
    df_clean["body"] = ""

    # Filling the title and body columns with the longer title and body
    df_clean.loc[df_clean["n3k_body_len"] > df_clean["bs_body_len"], "body"] = df_clean["n3k_body"]
    df_clean.loc[df_clean["n3k_body_len"] < df_clean["bs_body_len"], "body"] = df_clean["bs_body"]
    df_clean.loc[df_clean["n3k_title_len"] > df_clean["bs_title_len"], "title"] = df_clean["n3k_title"]
    df_clean.loc[df_clean["n3k_title_len"] < df_clean["bs_title_len"], "title"] = df_clean["bs_title"]

    df_clean["title_len"] = df_clean["title"].str.len()

    # Filling the title column with the se_title if longer than the title
    df_clean.loc[df_clean["se_title_len"] > df_clean["title_len"], "title"] = df_clean["se_title"]

    # Flag all instances of email, phone number, html_tags, and websites in all columns except the body with NAN for later removal
    for col in ["title", "paragraph", "description"]:
        df_clean[col] = df_clean[col].replace(removal_pattern, np.nan, regex=True)

    # Flag all instances of empty strings or only whitespace chars in all columns with NAN for later removal
    for col in ["title", "body", "paragraph", "description"]:
        df_clean[col] = df_clean[col].replace(empty_string_pattern, np.nan, regex=True)

    df_clean = df_clean[["article_index", "engine", "link", "source", "title", "description", "body", "paragraph"]]
    df_clean = df_clean.dropna(subset=["title", "description", "body", "paragraph"]).reset_index(drop=True)

    return df_clean
