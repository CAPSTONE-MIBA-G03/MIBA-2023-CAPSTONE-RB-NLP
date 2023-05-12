import pandas as pd
import argparse
import os

from datetime import datetime
from link_extractor import get_all_links
from content_extractor import get_content
from content_cleaner import clean_content


class PipelineExecutor:

    def __init__(self, main_dir='data', raw_dir='raw', clean_dir='clean') -> None:
        # assign variables
        self.main_dir = main_dir
        self.raw_dir = f'{main_dir}/{raw_dir}'
        self.clean_dir = f'{main_dir}/{clean_dir}'
        
        # setup storage
        self.__setup_storage()

    # helper methods
    def __setup_storage(self) -> None: 
        # parent data directory
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)

        # child data directory (raw)
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)

        # child data directory (clean)
        if not os.path.exists(self.clean_dir):
            os.mkdir(self.clean_dir)

    # main methods
    def execute(self, company, max_articles):
        links = get_all_links(company=company, max_articles=max_articles)

        links_df = pd.DataFrame(links)
        links_df = links_df.drop_duplicates(subset=["se_link"])
        links_df = links_df[~links_df['se_link'].isna()]

        links_list = links_df['se_link'].to_list()

        # 2. get content (and store)
        dirty_content_df = get_content(links_list)

        dirty_content_df = pd.DataFrame(dirty_content_df)
        dirty_content_df = pd.merge(links_df, dirty_content_df, left_on='se_link', right_on='bs_link')

        filename = f'{self.raw_dir}/{company.strip().replace(" ", "")}_{datetime.now()}.csv'
        dirty_content_df.to_csv(filename, index=False)

        # 3. clean content (and store)
        clean_content_df = clean_content(dirty_content_df)
        filename = f'{self.clean_dir}/{company.strip().replace(" ", "")}_{datetime.now()}.csv'
        clean_content_df.to_csv(filename, index=False)


# run only as main program
if __name__ == '__main__':

    # read command line arguments
    parser = argparse.ArgumentParser(
                        prog='PipelineExecutor',
                        description='Script to execute article extraction pipeline')

    parser.add_argument('-c', '--company', default='Credit Suisse')
    parser.add_argument('-x', '--max-articles', type=int, default=1)

    args = parser.parse_args()

    # execute pipeline
    pipe = PipelineExecutor()
    pipe.execute(args.company, args.max_articles)

