def clean_content(df_dirty):
    # Links are always identical here, so we can drop two of them and rename the remaining one
    df_dirty.drop(['n3k_link', 'bs_link'], axis=1, inplace=True)
    df_dirty.rename(columns={'se_link': 'link'}, inplace=True)

    # Removing all '\n' and '\t' from n3k_title, n3k_body, bs_title, bs_body as well as leading and trailing whitespaces and more than
    # one whitespace in between words
    for col in ['n3k_title', 'n3k_body', 'bs_title', 'bs_body']:
        df_dirty[col] = df_dirty[col].str.replace('\n|\t', ' ', regex = False).str.replace(' +', ' ', regex = True).str.strip()

    # replacing n3k_titles with NA if they contain certain words and are less than 20 characters
    for word in ['robot', 'subscribe', 'register']:
        df_dirty.loc[(df_dirty['n3k_title'].str.lower().str.contains(word))
                    & (df_dirty['n3k_title'].str.len() < 20), 'n3k_title'] = ''
        
    # doing the same with n3k_body
    for word in ['cookies', 'javascript', 'register', 'explorer', 'benzinga', 'djreprints']:
        df_dirty.loc[(df_dirty['n3k_body'].str.lower().str.contains(word))
                    & (df_dirty['n3k_body'].str.len() < 400), 'n3k_body'] = ''

    # Doing the same for bs_title and bs_body
    for word in ['yahoo finance', 'bloomberg', 'yahoo news', 'navigation', 'the straits times']:
        df_dirty.loc[(df_dirty['bs_title'].str.lower().str.contains(word))
                    & (df_dirty['bs_title'].str.len() < 20), 'bs_title'] = ''

    for word in ['javascript', 'copyright', 'benzinga', 'djreprints']:
        df_dirty.loc[(df_dirty['bs_body'].str.lower().str.contains(word))
                    & (df_dirty['bs_body'].str.len() < 400), 'bs_body'] = ''

    df_clean = df_dirty.copy()

    # Creating a column with the length of the n3k_body and bs_body
    df_clean['n3k_body_len'] = df_clean['n3k_body'].str.len()
    df_clean['bs_body_len'] = df_clean['bs_body'].str.len()

    # Creating a column with the length of the n3k_title, bs_title and se_title
    df_clean['n3k_title_len'] = df_clean['n3k_title'].str.len()
    df_clean['bs_title_len'] = df_clean['bs_title'].str.len()
    df_clean['se_title_len'] = df_clean['se_title'].str.len()

    # Creating and empty column for title and body
    df_clean['title'] = ''
    df_clean['body'] = ''

    # Filling the title and body columns with the longer title and body
    df_clean.loc[df_clean['n3k_body_len'] > df_clean['bs_body_len'], 'body'] = df_clean['n3k_body']
    df_clean.loc[df_clean['n3k_body_len'] < df_clean['bs_body_len'], 'body'] = df_clean['bs_body']
    df_clean.loc[df_clean['n3k_title_len'] > df_clean['bs_title_len'], 'title'] = df_clean['n3k_title']
    df_clean.loc[df_clean['n3k_title_len'] < df_clean['bs_title_len'], 'title'] = df_clean['bs_title']

    # creating a column with the length of the title
    df_clean['title_len'] = df_clean['title'].str.len()

    # Filling the title and body columns with the se_title if they are longer than the title
    df_clean.loc[df_clean['se_title_len'] > df_clean['title_len'], 'title'] = df_clean['se_title']

    # Dropping the columns that are not needed anymore
    df_clean.drop(['n3k_title', 'n3k_body', 'bs_title', 'bs_body', 'n3k_body_len', 'bs_body_len',
                'n3k_title_len', 'bs_title_len', 'se_title_len', 'se_title', 'title_len'], axis=1, inplace=True)

    # Dropping all rows that have no title or body
    df_clean.dropna(subset=['title', 'body'], inplace=True)

    return df_clean