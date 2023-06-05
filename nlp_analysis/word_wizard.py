import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
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
        If True, aims to use a smaller BERT model (bert-base-cased) instead of the default (bert-large-cased).

    Examples
    --------
    >>> from nlp_analysis import WordWizard
    >>> pipe = WordWizard(df = df)
    """

    def __init__(self, df, lean=True) -> None:
        self.df = df.copy().reset_index(drop=True)
        # Setup device agnostic code (Chooses NVIDIA (most optimized) or Metal backend (still better than cpu) if available, otherwise defaults to CPU)
        if torch.cuda.is_available():
            self.device = "cuda"

        elif torch.backends.mps.is_available():
            self.device = "mps"

        else:
            self.device = "cpu"

        # This code block should be in the ETL pipeline NOT in this nlp pipe
        self.df["paragraph"] = self.df["body"].str.split("\n\n")
        self.df = self.df.explode("paragraph", ignore_index=False)
        self.df = self.df.reset_index(names="para_index")
        self.df["sentences"] = self.df["paragraph"].apply(lambda x: sent_tokenize(x))

    def create_word_embeddings(self, columns: list([str]), lean=True, device=None):
        if device:
            self.device = device

        if lean:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            model = BertModel.from_pretrained("bert-large-cased", output_hidden_states=True)

        model.to(self.device)
        model.eval()

        for column in tqdm(columns, desc=f"Creating embeddings for column: {columns}", leave=True):
            new_column_name = column + "_word_embeddings"
            self.df[new_column_name] = None

            texts = self.df[column].tolist()
            for i, text in enumerate(tqdm(texts, leave=False)):
                if (i != 0) and (self.df.at[i, "para_index"] == self.df.at[i - 1, "para_index"]):
                    self.df.at[i, new_column_name] = self.df.at[i - 1, new_column_name]
                    continue

                encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                encoded_input.to(self.device)

                with torch.inference_mode():
                    outputs = model(**encoded_input)
                    last_hidden_state = outputs.last_hidden_state
                    embedding = torch.mean(last_hidden_state, dim=1).squeeze(0)
                    self.df.at[i, new_column_name] = embedding.cpu().numpy()

        return self.df

    def create_sentence_embeddings(self, device=None):
        # Testing for now shows that cpu is faster than gpu for this task, thus added device option
        if device:
            self.device = device

        model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        self.df["body_sentence_embeddings"] = None

        paragraphs = self.df["sentences"].tolist()
        for i, sentences in enumerate(tqdm(paragraphs, desc="Embedding body sentences", leave=False)):
            embedding = model.encode(sentences, device=self.device)
            self.df.at[i, "body_sentence_embeddings"] = embedding.mean(axis=0)

        return self.df

    def cluster_embeddings(self, column):
        pass

    def find_medoids(self, column):
        pass

    def find_sentiment(self, column):
        pass

    def entitiy_recognition(self, column):
        pass

    def topic_modelling(self, column):
        pass
