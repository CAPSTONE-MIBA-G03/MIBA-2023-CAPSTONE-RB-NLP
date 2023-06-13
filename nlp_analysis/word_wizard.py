import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
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

    # Class variables
    EMB_SUFFIX = '_word_embeddings'
    SENT_EMB_SUFFIX = '_sentence_embeddings'
    CLUSTER_SUFFIX = '_cluster_'
    SENTIMENT_SUFFIX = '_sentiment'
    NER_SUFFIX = '_NER'
    MEDOID_SUFFIX = '_is_medoid'

    def __init__(self, df) -> None:
        self.df = df.copy().reset_index(drop=True)
        # Setup device agnostic code (Chooses NVIDIA (most optimized) or Metal backend (still better than cpu) if available, otherwise defaults to CPU)
        if torch.cuda.is_available():
            self.device = "cuda"

        elif torch.backends.mps.is_available():
            self.device = "mps"

        else:
            self.device = "cpu"

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
            new_column_name = column + self.EMB_SUFFIX
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

    def cluster_embeddings(self, column, k_upperbound=15, extra_clusters=1, method='silhouette'):
        # Define main variables
        kmeans = {}
        K = range(2, k_upperbound)
        column = column + self.EMB_SUFFIX

        # Determine optimal K
        if method == 'silhouette':
            sil = []

            # Calculate Silhouette Score for each K
            for k in K:
                kmeans[k] = KMeans(n_clusters=k, n_init='auto').fit(self.df[column].tolist())
                labels = kmeans[k].labels_
                sil.append(silhouette_score(self.df[column].tolist(), labels, metric='euclidean'))

            # Find optimal K
            optimal_k = sil.index(max(sil)) + 2  # +2 because index starts from 0 and k starts from 2
        
        elif method == 'elbow':
            # ssd = sum of squared distances
            ssd = []

            # Calculate sum of squared distances for each K
            for k in K:
                model = KMeans(n_clusters=k, n_init='auto').fit(self.df[column].tolist())
                ssd.append(model.inertia_)

            # Plot sum of squared distances
            plt.plot(K, ssd, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum of squared distances')
            plt.title('Elbow Method For Optimal k')
            plt.show()

            # Ask user for optimal K
            try:
                optimal_k = int(input("Enter the optimal number of clusters (based on the plot): "))
            except ValueError:
                raise ValueError("Invalid input. Please enter an integer.")
       
        else:
            raise ValueError("Invalid method. Choose 'silhouette' or 'elbow'.")

        print(f"Optimal K: {optimal_k}")

        # Cluster with optimal K and extra_clusters
        n_clusters = range(max(2, optimal_k - extra_clusters), optimal_k + extra_clusters + 1)  # adding/subtracting extra_clusters

        # Add clusters to dataframe
        for k in n_clusters:

            # Train KMeans model if not already trained
            if k not in kmeans:
                kmeans[k] = KMeans(n_clusters=k, n_init='auto').fit(self.df[column].tolist())

            # Add cluster labels to dataframe
            new_column = column + self.CLUSTER_SUFFIX + str(k)
            self.df[new_column] = kmeans[k].labels_
            
            # Finding Medoids (hard to implement as standalone method because kmeans is instantiated in this method)
            centroids = kmeans[k].cluster_centers_
            closest_medoid_indices, _ = pairwise_distances_argmin_min(self.df[column].tolist(), centroids)
            self.df[new_column + self.MEDOID_SUFFIX] = False
            self.df.loc[closest_medoid_indices, new_column + "_is_medoid"] = True        

        return self


    def find_sentiment(self, columns: list([str]), device=None):

        if device:
            self.device = device

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        model.to(self.device)
        model.eval()

        for column in tqdm(columns, desc=f"Calculating Sentiment for column(s): {columns}", leave=True):
            new_column_name = column + self.SENTIMENT_SUFFIX
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
            new_column_name = column + self.NER_SUFFIX
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
