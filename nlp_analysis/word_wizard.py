import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from tqdm.auto import tqdm
import spacy

tqdm.pandas()
import transformers
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
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

    device : str, optional
        The device to use for the analysis. If not specified, the class will automatically choose the best possible device available.

    Examples
    --------
    >>> from nlp_analysis import WordWizard
    >>> pipe = WordWizard(df = df)
    """

    # Class variables
    EMB_SUFFIX = '_word_embeddings'
    SENT_EMB_SUFFIX = '_sentence_embeddings'
    CLUSTER_SUFFIX = '_clusters'
    SENTIMENT_SUFFIX = '_sentiment'
    NER_SUFFIX = '_NER'
    MEDOID_SUFFIX = '_medoids'
    SUMMARY_SUFFIX = '_summaries'
    REDDIM_SUFFIX = '_reduced_dimensions'

    def __init__(self, df, device=None, interest = 'paragraph') -> None:
        self.df = df.copy().reset_index(drop=True)
        # Setup device agnostic code (Chooses NVIDIA (most optimized) or Metal backend (still better than cpu) if available, otherwise defaults to CPU)
        if device:
            self.device = device

        elif torch.cuda.is_available():
            self.device = "cuda"

        elif torch.backends.mps.is_available():
            self.device = "mps"

        else:
            self.device = "cpu"
        
        if interest == 'body':
            self.df = self.df.drop(columns=['article_index', 'paragraph'])
            self.df = self.df.drop_duplicates()
            self.df = self.df.reset_index(drop=True) # could make problems somewhere
            self.interest = interest
        


        # delete
        elif interest == 'skip':
            pass




        else:
            self.df["sentences"] = self.df["paragraph"].apply(lambda x: sent_tokenize(x))
            self.interest = interest

    def create_word_embeddings(self, column: str, lean=True, device=None):
        if not device:
            device = self.device

        if lean:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
            model = BertModel.from_pretrained("bert-large-cased", output_hidden_states=True)

        model.to(device)
        model.eval()
        
        new_column_name = column + self.EMB_SUFFIX # e.g.: paragraph_word_embeddings
        self.df[new_column_name] = None
        for i, entry in enumerate(tqdm(self.df[column], desc=f"Creating word embeddings for {column}")):
            encoded_input = tokenizer(entry, padding=True, truncation=True, return_tensors="pt")
            encoded_input.to(device)

            with torch.inference_mode():
                outputs = model(**encoded_input)
                last_hidden_state = outputs.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1).squeeze(0).cpu().numpy()

            self.df.at[i, new_column_name] = embedding
        
        return self
    
        unique_indices = self.df[~self.df[column].duplicated()].index.tolist()
        for i, pos in enumerate(tqdm(unique_indices, desc=f"Creating word embeddings for column {column}")):

            text = self.df.at[pos, column]

            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            encoded_input.to(device)

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

    def create_sentence_embeddings(self, column="sentences" ,device=None):
        # Testing for now shows that cpu is faster than gpu for this task, thus added device option
        if not device:
            device = self.device

        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.df[column + self.SENT_EMB_SUFFIX] = model.encode(self.df[column].tolist(), show_progress_bar=True).tolist()

        return self

    def cluster_embeddings(self, column, k_upperbound=15, extra_clusters=1, method='silhouette', k=None, n_med=2):
        df = self.df
        embed_column = self._get_embed_col(column)

        # Define main variables
        kmeans = {}
        K = range(2, k_upperbound)
        
        if (k == None) & (method == 'silhouette'):
            sil = []

            # Calculate Silhouette Score for each K
            for k in K:
                kmeans[k] = KMeans(n_clusters=k, n_init='auto').fit(df[embed_column].tolist())
                labels = kmeans[k].labels_
                sil.append(silhouette_score(df[embed_column].tolist(), labels))

            # Find optimal K
            k = sil.index(max(sil)) + 2  # +2 because index starts from 0 and k starts from 2
        
        elif (k == None) & (method == 'elbow'):
            print('hi')
            # ssd = sum of squared distances
            ssd = []

            # Calculate sum of squared distances for each K
            for k in K:
                model = KMeans(n_clusters=k, n_init='auto').fit(df[embed_column].tolist())
                ssd.append(model.inertia_)

            # Plot sum of squared distances
            plt.plot(K, ssd, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum of squared distances')
            plt.title('Elbow Method For Optimal k')
            plt.show()

            # Ask user for optimal K
            try:
                k = int(input("Enter the optimal number of clusters (based on the plot): "))
            except ValueError:
                raise ValueError("Invalid input. Please enter an integer.")

        kmeans = KMeans(n_clusters=k, n_init='auto').fit(df[embed_column].tolist())
        clust_col = embed_column + self.CLUSTER_SUFFIX 
        
        # Add cluster labels to dataframe
        df[clust_col] = kmeans.labels_
        
        # Finding Medoids
        centroids = kmeans.cluster_centers_
        df[clust_col + self.MEDOID_SUFFIX] = False
        for i, centroid in enumerate(centroids):
            points_in_cluster = df[df[clust_col] == i][embed_column]
            distances = points_in_cluster.apply(lambda x: np.linalg.norm(np.array(x) - centroid))
            closest_indices = distances.nsmallest(n_med).index
            df.loc[closest_indices, clust_col + self.MEDOID_SUFFIX] = True

    def summarize_medoids(self, column: str, lean=True, device=None):

        if not device:
            device = self.device

        if lean:
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
        else:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

        model.to(device)

        new_column_name = column + self.MEDOID_SUFFIX + self.SUMMARY_SUFFIX
        
        if column == 'sentences':
            medoid = column + self.SENT_EMB_SUFFIX + self.CLUSTER_SUFFIX + self.MEDOID_SUFFIX
            new_column_name = column + self.SENT_EMB_SUFFIX + self.CLUSTER_SUFFIX + self.MEDOID_SUFFIX + self.SUMMARY_SUFFIX
        else:
            medoid = column + self.EMB_SUFFIX + self.CLUSTER_SUFFIX + self.MEDOID_SUFFIX
            new_column_name = column + self.EMB_SUFFIX + self.CLUSTER_SUFFIX + self.MEDOID_SUFFIX + self.SUMMARY_SUFFIX

        self.df[new_column_name] = np.nan

        medoid_indices = self.df.loc[self.df[medoid] == True].index.tolist()
        for i, pos in enumerate(tqdm(medoid_indices, desc=f"Creating summaries for medoids of column {column}")):

            text = self.df.at[pos, column]

            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            encoded_input.to(device)

            with torch.inference_mode():
                summary = model.generate(**encoded_input)
                out = tokenizer.decode(summary[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            self.df.at[pos, new_column_name] = out

        return self

    def find_sentiment(self, column: str, device=None):
        """
        Computes the sentiment score for the input column and adds a new column with the suffix '_sentiment.'

        Parameters
        ----------
        column : {"title", "description", "body", "paragraphs", "sentences"}
            The column to find the sentiment score for.

        device : str, optional
            The device to use for the model. If not specified, the default device is used.

        Examples
        --------
        >>> from nlp_analysis import WordWizard
        >>> pipe = WordWizard(df = df)
        >>> pipe.find_sentiment(column = "body")
        >>> pipe.df["body_sentiment"].head()
        """

        if not device:
            device = self.device

        tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
        model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

        model.to(device)
        model.eval()

        new_column_name = column + self.SENTIMENT_SUFFIX
        self.df[new_column_name] = np.nan

        unique_indices = self.df[~self.df[column].duplicated()].index.tolist()
        for i, pos in enumerate(tqdm(unique_indices, desc=f"Calculating sentiment for column {column}")):

            text = self.df.at[pos, column]

            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            encoded_input.to(device)

            with torch.inference_mode():
                outputs = model(**encoded_input).logits
                pred = torch.argmax(outputs).item()
            
            if i + 1 == len(unique_indices):
                self.df.loc[pos:, new_column_name] = self.df.loc[pos:].apply(lambda _: pred, axis=1)

            else:
                next_index = unique_indices[i + 1]
                self.df.loc[pos:next_index - 1, new_column_name] = self.df.loc[pos:next_index - 1].apply(lambda _: pred, axis=1)

        return self
    
    def entitiy_recognition(self, column: str, lean=True, device=None, w_t=3, w_d=2, top_n=5):
        """
        extract the top_n most frequently appearing organizations in a text
        :param text: text to extract organizations from
        :param w1: weight for title
        :param w2: weight for description
        :param top_n: number of organizations to return
        :return: list of top_n organizations
        """

        if not device:
            device = self.device
        
        if lean:
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        else:
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

        nlp = spacy.load("en_core_web_lg")

        # adding a pattern that will not classify quantum (capital or lower), AI, Company, quantum computing as organizations
        patterns = [{'label': 'NORG', 'pattern': 'Quantum'},
                    {'label': 'NORG', 'pattern': 'quantum'},
                    {'label': 'NORG', 'pattern': 'AI'},
                    {'label': 'NORG', 'pattern': 'Company'},
                    {'label': 'NORG', 'pattern': 'quantum computing'},
                    {'label': 'NQORG', 'pattern': 'NYSE'}, # not quantum orgs
                    {'label': 'NQORG', 'pattern': 'NASDAQ'}]
        
        # adding the patterns to the nlp model, if it is not already there
        if 'entity_ruler' not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(patterns)
        
        def get_organizations(text):
            doc = nlp(text)
            orgs = [str(ent) for ent in doc.ents if ent.label_ == 'ORG']
            return orgs

        
        self.df[column + self.CLUSTER_SUFFIX + self.EMB_SUFFIX + self.NER_SUFFIX] = None
        unique_clusters = self.df[column + self.EMB_SUFFIX + self.CLUSTER_SUFFIX].unique()
        # using tqdm on for i in unique_clusters:
        for i in tqdm(unique_clusters, desc=f"Extracting organizations for column {column}"):
            sub = self.df[self.df[column + self.EMB_SUFFIX + self.CLUSTER_SUFFIX] == i]
            
            # Adding a column with organizations for each title, body and description
            for col in ['title', 'description', column]:
                sub = sub.assign(**{col+'_orgs': sub[col].apply(lambda x: get_organizations(x) if type(x) == str else [])})

            # getting the n most important organizations for each cluster
            all_orgs = sub['title_orgs'].explode().tolist() * w_t + sub['description'].explode().tolist() * w_d + sub[column + '_orgs'].explode().tolist()
            all_orgs = [org for org in all_orgs if type(org) != float]
            orgs = Counter(all_orgs)
            orgs_list = [org[0] for org in orgs.most_common(top_n)]
            self.df.loc[(self.df[column + self.EMB_SUFFIX + self.CLUSTER_SUFFIX] == i), column + self.CLUSTER_SUFFIX + self.EMB_SUFFIX + self.NER_SUFFIX] = str(orgs_list)

        return self
    
    
        if not device:
            device = self.device
        
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

    def reduce_demensionality(self, column, n_components=2, n_neighbors=15, min_dist=0.0, metric='cosine'):
        # check if either word_embeddings or sentence embeddings are present else raise error
        embed_column = self._get_embed_col(column)
        
        # getting the embeddings for the column
        embeddings = self.df[embed_column].tolist()

        # creating the umap embeddings
        umap_data = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=metric).fit_transform(embeddings)

        # adding the umap embeddings to the dataframe
        self.df[column + self.REDDIM_SUFFIX + self.EMB_SUFFIX] = umap_data.tolist()

        return self

    def topic_modelling(self, column, with_embeddings=True, sample_size=1000):

        # checking if clusters are already created
        cluster_col = self._get_cluster_col(column)

        # create c-tf-idf helper function
        def c_tf_idf(documents, m, ngram_range=(1, 1)):
            count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
            t = count.transform(documents).toarray()
            w = t.sum(axis=1)
            tf = np.divide(t.T, w)
            sum_t = t.sum(axis=0)
            idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
            tf_idf = np.multiply(tf, idf)

            return tf_idf, count
        
        # create function to extract top n words per topic
        def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
            words = count.get_feature_names_out()
            labels = list(docs_per_topic[cluster_col])
            tf_idf_transposed = tf_idf.T
            indices = tf_idf_transposed.argsort()[:, -n:]
            top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
            return top_n_words

        # create docs_df with the column and the cluster
        docs_df = self.df[[column, cluster_col]].copy()

        # create docs per topic
        docs_per_topic = docs_df.groupby([cluster_col], as_index = False).agg({column: ' '.join})

        # get tf-idf and count vectorizer
        tf_idf, count = c_tf_idf(docs_per_topic[column].values, len(self.df))
        
        return extract_top_n_words_per_topic(tf_idf, count, docs_per_topic)
    
    # Helper functions
    def _get_embed_col(self, column):

        # check if either word_embeddings or sentence embeddings are present else raise error
        if column + self.EMB_SUFFIX in self.df.columns:
            return column + self.EMB_SUFFIX
        
        elif column + self.SENT_EMB_SUFFIX in self.df.columns:
            return column + self.SENT_EMB_SUFFIX
        
        else:
            raise ValueError(f"Column {column} does not exist in dataframe. Please create embeddings first.")
        

    def _get_cluster_col(self, column):
        embed_column = self._get_embed_col(column)

        if embed_column + self.CLUSTER_SUFFIX in self.df.columns:
            return embed_column + self.CLUSTER_SUFFIX
        
        else:
            raise ValueError(f"Column {column} does not exist in dataframe. Please create clusters first.")
