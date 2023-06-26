import argparse
import os

import pandas as pd

from etl_pipeline.content_cleaner import clean_content
from etl_pipeline.content_extractor import get_content
from etl_pipeline.link_extractor import get_all_links


class PipelineExecutor:
    """
    Class that provides an interface to execute the entire ETL pipeline for a given query.

    The pipeline consists of the following steps:
    1. get links
    2. get content (and return/store)
    3. clean content (and return/store)

    Attributes
    ----------
    main_dir : str
        The main directory for the pipeline

    raw_dir : str
        The raw data directory for the pipeline

    clean_dir : str
        The clean data directory for the pipeline

    Methods
    -------
    execute(query, max_articles=None, overwrite=False)
        Executes ETL pipeline for a given query and returns the clean content as a dataframe.

    Examples
    --------
    >>> from pipeline_executor import PipelineExecutor
    >>> pipe = PipelineExecutor()
    >>> pipe.execute(query="Roland Berger", max_articles=50)

    """

    def __init__(self, main_dir="data", raw_dir="raw", clean_dir="clean") -> None:
        # assign variables
        self.main_dir = main_dir
        self.raw_dir = f"{main_dir}/{raw_dir}"
        self.clean_dir = f"{main_dir}/{clean_dir}"

        # setup storage
        self.__setup_storage()

    # helper methods
    def __setup_storage(self) -> None:
        """
        Sets up the storage for the pipeline
        """

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
    def execute(self, query, max_articles=None, overwrite=False):
        """
        Executes ETL pipeline for a given query and returns the clean content as a dataframe.

        Parameters
        ----------
        query : str
            The news query to be searched for.

        max_articles : int
            The maximum number of articles to extract for each Search Engine. Extracts all articles by default.

        overwrite : bool
            Whether to overwrite existing files or not. Defaults to False.

        Returns
        -------
        clean_content_df : pd.DataFrame
            The clean content as a dataframe, ready for analysis using the WordWizard class.
        """

        # filenames
        build_filename = lambda dir: f'{dir}/{query.strip().replace(" ", "")}_{max_articles}.csv'
        raw_filename = build_filename(self.raw_dir)
        clean_filename = build_filename(self.clean_dir)

        # return if file with [query] already exists
        if os.path.exists(clean_filename) and not overwrite:
            return pd.read_csv(clean_filename)

        # 1. get links
        links = get_all_links(query=query, max_articles=max_articles)

        links_df = pd.DataFrame(links)
        links_df = links_df[~links_df["se_link"].isna()]

        links_list = links_df["se_link"].to_list()

        # 2. get content (and store)
        dirty_content_df = get_content(links_list)

        dirty_content_df = pd.DataFrame(dirty_content_df)
        dirty_content_df = pd.merge(links_df, dirty_content_df, left_on="se_link", right_on="bs_link")
        dirty_content_df = dirty_content_df.explode("bs_paragraph", ignore_index=False)
        dirty_content_df = dirty_content_df.reset_index(names="article_index")

        dirty_content_df.to_csv(raw_filename, index=False)

        # 3. clean content (and store)
        clean_content_df = clean_content(dirty_content_df)
        clean_content_df.to_csv(clean_filename, index=False)

        # return clean content
        return clean_content_df


# run only as main program
if __name__ == "__main__":
    # read command line arguments
    parser = argparse.ArgumentParser(
        prog="PipelineExecutor", description="Script to execute article extraction pipeline"
    )

    parser.add_argument("-q", "--query", default="Roland Berger")
    parser.add_argument("-x", "--max-articles", type=int, default=None)

    args = parser.parse_args()

    # execute pipeline
    pipe = PipelineExecutor()
    pipe.execute(args.query, args.max_articles)
