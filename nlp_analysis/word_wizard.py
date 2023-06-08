import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          BartModel, BartTokenizer, BertModel, BertTokenizer,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer, pipeline)


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

    def __init__(self, df) -> None:
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

        for column in tqdm(columns, desc=f"Creating embeddings for column(s): {columns}", leave=True):
            new_column_name = column + "_word_embeddings"
            self.df[new_column_name] = None

            unique_indices = self.df[~self.df[column].duplicated()].index.tolist()
            for i, pos in enumerate(tqdm(unique_indices, leave=False)):

                text = self.df.at[pos, column]

                encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                encoded_input.to(self.device)

                with torch.inference_mode():
                    outputs = model(**encoded_input)
                    last_hidden_state = outputs.last_hidden_state
                    embedding = torch.mean(last_hidden_state, dim=1).squeeze(0).cpu().numpy()
                
                if i + 1 == len(unique_indices):
                    self.df.loc[pos:, new_column_name] = self.df.loc[pos:].apply(lambda _: embedding.tolist(), axis=1)

                else:
                    next_index = unique_indices[i + 1]
                    self.df.loc[pos:next_index - 1, new_column_name] = self.df.loc[pos:next_index - 1].apply(lambda _: embedding.tolist(), axis=1)

        return self

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

        return self

    def cluster_embeddings(self, column, k_upperbound=15, extra_clusters=1):
        '''
        Clusters the word embeddings of a column in the dataframe.
        
        Parameters
        ----------
        column : str
            The column name of the dataframe to cluster.
        k_upperbound : int
            The upperbound of the range of k values to test for the optimal number of clusters.
        extra_clusters : int
            The number of extra clusters to add/subtract from the optimal number of clusters.

        Examples
        --------
        >>> from nlp_analysis import WordWizard
        >>> pipe = WordWizard(df = df)
        >>> pipe.create_word_embeddings(columns = ["body"])
        >>> pipe.cluster_embeddings(column = "body_word_embeddings")
        >>> pipe.df.head()

        Notes
        -----
        This method uses the elbow method to find the optimal number of clusters.

        References
        ----------
        https://www.kaggle.com/sonalidasgupta/clustering-using-bert-embeddings
        '''

        sil = []
        K = range(2, k_upperbound)
        for k in K:
            kmeans = KMeans(n_clusters=k).fit(self.df[column].tolist())
            labels = kmeans.labels_
            sil.append(silhouette_score(self.df[column].tolist(), labels, metric='euclidean'))

        optimal_k = sil.index(max(sil)) + 2  # +2 because index starts from 0 and k starts from 2

        n_clusters = range(max(2, optimal_k - extra_clusters), optimal_k + extra_clusters + 1)  # adding/subtracting extra_clusters
        for n in n_clusters:
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(self.df[column].tolist())
            self.df[column + 'cluster' + str(n)] = kmeans.labels_

        return self

    def find_medoids(self, columns):
        pass

    def find_sentiment(self, columns: list([str]), device=None):

        if device:
            self.device = device

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        model.to(self.device)
        model.eval()

        for column in tqdm(columns, desc=f"Calculating Sentiment for column(s): {columns}", leave=True):
            new_column_name = column + "_sentiment"
            self.df[new_column_name] = None

            texts = self.df[column].tolist()
            for i, text in enumerate(tqdm(texts, leave=False)):
                if (i != 0) and (self.df.at[i, "para_index"] == self.df.at[i - 1, "para_index"]) and (column != "paragraph"):
                    self.df.at[i, new_column_name] = self.df.at[i - 1, new_column_name]
                    continue

                encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                encoded_input.to(self.device)

                with torch.inference_mode():
                    logits = model(**encoded_input).logits
                    sentiment = logits.argmax().item()
                    self.df.at[i, new_column_name] = sentiment

        return self
    
    def entitiy_recognition(self, columns: list([str]), lean=True, device=None):
        if device:
            self.device = device
        
        if lean:
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        else:
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

        for column in tqdm(columns, desc=f"Creating embeddings for column: {columns}", leave=True):
            new_column_name = column + "_NER"
            self.df[new_column_name] = None

            texts = self.df[column].tolist()
            for i, text in enumerate(tqdm(texts, leave=False)):
                if (i != 0) and (self.df.at[i, "para_index"] == self.df.at[i - 1, "para_index"]) and (column != "paragraph"):
                    self.df.at[i, new_column_name] = self.df.at[i - 1, new_column_name]
                    continue

                pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=self.device)
                self.df.at[i, new_column_name] = pipe(text)

        return self

    def topic_modelling(self, column):
        pass
