import pandas as pd
import argparse
import os

from datetime import datetime
from link_extractor import get_all_links
from content_extractor import get_content


def setup_storage(data_dir: str, raw_dir: str, clean_dir: str) -> None: 
    # parent data directory
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # child data directory (raw)
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)
    
    # child data directory (clean)
    if not os.path.exists(clean_dir):
        os.mkdir(clean_dir)


def get_args():
    parser = argparse.ArgumentParser(
                        prog='PipelineExecutor',
                        description='Script to execute article extraction pipeline')

    parser.add_argument('-c', '--company', default='Credit Suisse')
    parser.add_argument('-x', '--max-articles', type=int, default=2)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()

    DATA_DIR = './data'
    RAW_DIR = f'{DATA_DIR}/raw'
    CLEAN_DIR = f'{DATA_DIR}/clean'

    # add directories if they don't exist
    setup_storage(DATA_DIR, RAW_DIR, CLEAN_DIR)

    # 1. get links
    links = get_all_links(company=args.company, max_articles=args.max_articles)

    links_df = pd.DataFrame(links)
    links_df = links_df.drop_duplicates(subset=["se_link"])
    links_df = links_df[~links_df['se_link'].isna()]

    links_list = links_df['se_link'].to_list()

    # 2. get content (and store)
    content = get_content(links_list)

    content_df = pd.DataFrame(content)
    content_df = pd.merge(links_df, content_df, left_on='se_link', right_on='bs_link')

    filename = f'{RAW_DIR}/{args.company.strip().replace(" ", "")}_{datetime.now()}.csv'
    content_df.to_csv(filename, index=False)

    # TODO Clean data (and store)
