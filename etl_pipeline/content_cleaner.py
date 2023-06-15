import numpy as np
import pandas as pd


def clean_content(df):

    df_dirty = df.copy()
 
    # Drop identical columns and rename relevant columns
    df_dirty.drop(['n3k_link', 'bs_link'], axis=1, inplace=True)
    df_dirty.rename(columns={'se_link': 'link', 'se_description': 'description', 'bs_paragraph': 'paragraph', 'se_source': 'source'}, inplace=True)

    # Regex patterns to detect websites, emails, phone numbers and empty strings
    website_pattern = r'(?:http[s]?://)?www\.[^\s.]+\.[^\s]{2,}|^https?:\/\/.*[\r\n]*'
    email_pattern = r'[\w.-]+@[\w.-]+\.[\w.-]+'
    phone_pattern = r'\+?\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    empty_string_pattern = r'^\s*$'

    removal_pattern = fr'\b(?:{website_pattern}|{email_pattern}|{phone_pattern})\b'

    # Replace all '\n' and '\t', multiple spaces, and leading and trailing spaces with a single space
    for col in ['n3k_title', 'n3k_body', 'bs_title', 'bs_body', 'paragraph', 'description']:
        df_dirty[col] = df_dirty[col].str.replace(r'\n|\t| +', ' ', regex=True).str.strip()

    # Replace too short entries or ones which contain unwanted words with empty strings for later removal
    for word in ['robot', 'subscribe', 'register']:
        df_dirty.loc[(df_dirty['n3k_title'].str.lower().str.contains(word))
                    | (df_dirty['n3k_title'].str.len() < 20), 'n3k_title'] = ''
        
    for word in ['cookies', 'javascript', 'register', 'explorer', 'benzinga', 'djreprints']:
        df_dirty.loc[(df_dirty['n3k_body'].str.lower().str.contains(word))
                    | (df_dirty['n3k_body'].str.len() < 400), 'n3k_body'] = ''

    for word in ['yahoo finance', 'bloomberg', 'yahoo news', 'navigation', 'the straits times']:
        df_dirty.loc[(df_dirty['bs_title'].str.lower().str.contains(word))
                    | (df_dirty['bs_title'].str.len() < 20), 'bs_title'] = ''

    for word in ['javascript', 'copyright', 'benzinga', 'djreprints']:
        df_dirty.loc[(df_dirty['bs_body'].str.lower().str.contains(word))
                    | (df_dirty['bs_body'].str.len() < 400), 'bs_body'] = ''
        
    for word in ['javascript', 'copyright', 'benzinga', 'djreprints']:
        df_dirty.loc[(df_dirty['paragraph'].str.lower().str.contains(word))
                    | (df_dirty['paragraph'].str.len() < 150), 'paragraph'] = ''
        
    for word in ['javascript', 'copyright', 'benzinga', 'djreprints']:
        df_dirty.loc[(df_dirty['description'].str.lower().str.contains(word))
                    | (df_dirty['description'].str.len() < 100), 'description'] = ''
        
    # Flag all instances of email, phone number, and websites in all columns except the body with empty strings for later removal
    for col in ['n3k_title', 'bs_title', 'paragraph', 'description']:
        df_dirty[col] = df_dirty[col].replace(removal_pattern, '', regex=True)
        
    # Flag all instances of empty strings or only whitespace chars in all columns with np.nan for removal
    for col in ['n3k_title', 'bs_title', 'n3k_body', 'bs_body', 'paragraph', 'description']:
        df_dirty[col] = df_dirty[col].replace(empty_string_pattern, np.nan, regex=True)
    
    # Dropping all rows that have no title, description, body or paragraph
    df_clean = df_dirty.copy().dropna(subset=['n3k_title', 'bs_title', 'n3k_body', 'bs_body', 'paragraph', 'description'])

    # Creating a column with the length of the n3k_body and bs_body
    df_clean['n3k_body_len'] = df_clean['n3k_body'].str.len()
    df_clean['bs_body_len'] = df_clean['bs_body'].str.len()

    # Creating a column with the length of the n3k_title, bs_title and se_title
    df_clean['n3k_title_len'] = df_clean['n3k_title'].str.len()
    df_clean['bs_title_len'] = df_clean['bs_title'].str.len()
    df_clean['se_title_len'] = df_clean['se_title'].str.len()

    # Creating an empty column for title and body
    df_clean['title'] = ''
    df_clean['body'] = ''

    # Filling the title and body columns with the longer title and body
    df_clean.loc[df_clean['n3k_body_len'] > df_clean['bs_body_len'], 'body'] = df_clean['n3k_body']
    df_clean.loc[df_clean['n3k_body_len'] < df_clean['bs_body_len'], 'body'] = df_clean['bs_body']
    df_clean.loc[df_clean['n3k_title_len'] > df_clean['bs_title_len'], 'title'] = df_clean['n3k_title']
    df_clean.loc[df_clean['n3k_title_len'] < df_clean['bs_title_len'], 'title'] = df_clean['bs_title']

    # creating a column with the length of the title
    df_clean['title_len'] = df_clean['title'].str.len()

    # Filling the title column with the se_title if longer than the title
    df_clean.loc[df_clean['se_title_len'] > df_clean['title_len'], 'title'] = df_clean['se_title']

    return df_clean[['article_index', 'engine', 'link', 'source', 'title', 'description', 'body', 'paragraph']]
