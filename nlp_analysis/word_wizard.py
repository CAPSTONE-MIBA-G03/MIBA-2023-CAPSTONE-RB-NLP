import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class WordWizard:
    """
    A class for performing NLP analysis on a pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe containing text data.
    lean : bool
        If True, uses a smaller BERT model (bert-base-cased) instead of the default (bert-large-cased).

    Examples
    --------
    >>> from nlp_analysis import WordWizard
    >>> pipe = WordWizard(df = df)
    """

    def __init__(self, df, lean=False) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.lean = lean
        # Setup device agnostic code (Chooses NVIDIA or Metal backend if available, otherwise defaults to CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")

        else:
            self.device = torch.device("cpu")

    def create_embeddings(self, columns: list):
        if self.lean:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            model = BertModel.from_pretrained("bert-large-cased", output_hidden_states=True)

        model.to(self.device)
        model.eval()

        for column in tqdm(columns, desc=f"Creating embeddings for column: {columns}", position=0, leave=True):
            new_column_name = column + "_embedded"
            self.df[new_column_name] = None

            texts = self.df[column].tolist()
            for i, text in enumerate(tqdm(texts, leave=False, position=1)):
                encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                encoded_input.to(self.device)

                with torch.inference_mode():
                    outputs = model(**encoded_input)
                    last_hidden_state = outputs.last_hidden_state
                    embedding = torch.mean(last_hidden_state, dim=1).squeeze(0)
                    self.df.at[i, new_column_name] = embedding.cpu().numpy()

        return self.df

    def cluster_embeddings(self, column):
        pass

    def find_medoids(self, column):
        pass

    def find_sentiment(self, column):
        pass

    def entitiy_recognition(self, column):
        pass
