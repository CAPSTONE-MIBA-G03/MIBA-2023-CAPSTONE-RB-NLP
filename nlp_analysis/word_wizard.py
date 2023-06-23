import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from tqdm.auto import tqdm
from umap import UMAP

tqdm.pandas()
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          BertModel, BertTokenizer,
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

    interest : str, optional
        The news article specific attribute to analyze. Defaults to 'paragraph'.
        - 'paragraph': Paragraphs of the news article are used for analysis.
        - 'sentences': Sentences of the news article are used for analysis.

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
        # Setup nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

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
        
        # Setup interest (should expand to description and title as well) -> just design choice, code shoukd work for all
        # Further, might revisit 'lean' feature and implement it so that lean is good for cpu while set to false is advisable to use GPU
        # This is already implemented in the create_word_embeddings, sentiment, (and somewhat summarization - aka GPU is slightly broken there)

        if interest == 'body':
            self.df = self.df.drop(columns=['article_index', 'paragraph'])
            self.df = self.df.drop_duplicates()
            self.df = self.df.reset_index(drop=True) # could make problems somewhere
            self.interest = interest

        else:
            self.df["sentences"] = self.df["paragraph"].apply(lambda x: sent_tokenize(x))
            self.interest = interest

    def create_word_embeddings(self, lean=True, device=None):
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
        
        new_column_name = self.interest + self.EMB_SUFFIX # e.g.: paragraph_word_embeddings
        self.df[new_column_name] = None
        for i, entry in enumerate(tqdm(self.df[self.interest], desc=f"Creating word embeddings for {self.interest}")):
            encoded_input = tokenizer(entry, padding=True, truncation=True, return_tensors="pt")
            encoded_input.to(device)

            with torch.inference_mode():
                outputs = model(**encoded_input)
                last_hidden_state = outputs.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1).squeeze(0).cpu().numpy()

            self.df.at[i, new_column_name] = embedding
        
        return self

    def create_sentence_embeddings(self, device=None):

        if not device:
            device = self.device

        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.df[self.interest + self.SENT_EMB_SUFFIX] = model.encode(self.df[self.interest].tolist(), show_progress_bar=True).tolist()

        return self

    def cluster_embeddings(self, algorithm='kmeans', method='silhouette', k=None, k_max=15, k_min=5, n_med=2):
        """
        Clusters the word embeddings of the specified column.

        Parameters
        ----------
        column : str
            The column to cluster.
        algorithm : str, optional
            The clustering algorithm to use. Defaults to 'kmeans'.
            - 'kmeans': K-Means clustering.
            - 'hdbscan': HDBSCAN clustering.
        method : str, optional
            The method to use for determining the optimal number of clusters. Defaults to 'silhouette'.
        k : int, optional
            The number of clusters to use. If specified, the algorithm parameter will be ignored. Defaults to None.
        k_max : int, optional
            The upperbound for the number of clusters to try. Defaults to 15.
        k_min : int, optional
            The minimum number of clusters to try. Defaults to 5.
        n_med : int, optional
            The number of medoids to find. Defaults to 2.

        Returns
        -------
        self : WordWizard
            The WordWizard object.

        """

        df = self.df
        embed_column = self._get_embed_col(self.interest)
        clust_column = embed_column + self.CLUSTER_SUFFIX

        # Cluster
        if algorithm == 'kmeans':
            model = self._kmeans_clustering(df, embed_column, k_max, method, k, k_min)
            df[clust_column] = model.labels_

        elif algorithm == 'hdbscan':
            reduced_column = self.interest + self.REDDIM_SUFFIX + self.EMB_SUFFIX
            
            if reduced_column not in df.columns:
                self.reduce_demensionality(self.interest)
            
            model = HDBSCAN(min_cluster_size=k_min).fit(df[self.interest + self.REDDIM_SUFFIX + self.EMB_SUFFIX].tolist())
            df[clust_column] = model.labels_

        else:
            raise ValueError('Invalid algorithm. Choose either "kmeans" or "hdbscan".')

        # Find medoids
        self._find_medoids(model, n_med, algorithm)
        
        return self

    def summarize_medoids(self, lean=True, device=None):
        
        cluster_col = self._get_cluster_col(self.interest)
        
        if not device:
            device = self.device

        if lean:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            device = "cpu" # MPS bug currently doesnt allow GPU for this model (its fast on cpu anyways though)

        else:
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

        model.to(device)
        
        medoid = cluster_col + self.MEDOID_SUFFIX
        new_column_name = cluster_col + self.MEDOID_SUFFIX + self.SUMMARY_SUFFIX

        self.df[new_column_name] = np.nan

        medoid_indices = self.df.loc[self.df[medoid] == True].index.tolist()
        for _, pos in enumerate(tqdm(medoid_indices, desc=f"Creating summaries for cluster medoids based on {self.interest}")):

            text = self.df.at[pos, self.interest]

            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            encoded_input.to(device)

            with torch.inference_mode():
                summary = model.generate(**encoded_input)
                out = tokenizer.decode(summary[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            self.df.at[pos, new_column_name] = out

        return self

    def find_sentiment(self, lean=True, device=None):
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

        # What if we calculate sentiment on medoids as well (similar to summarization)? -> NewsSentiment 1.1.25 library exists as well
        # Might need more specific sentiment model trained on news (not 100% sure tho)

        if not device:
            device = self.device

        if lean:
            tokenizer = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
            model = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset") # 3 classes: positive(2), neutral(1), negative(0)

        else:
            tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
            model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

        model.to(device)
        model.eval()

        new_column_name = self.interest + self.SENTIMENT_SUFFIX
        self.df[new_column_name] = np.nan

        unique_indices = self.df[~self.df[self.interest].duplicated()].index.tolist()
        for i, pos in enumerate(tqdm(unique_indices, desc=f"Calculating sentiment for {self.interest}")):

            text = self.df.at[pos, self.interest]

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
    
    def entitiy_recognition(self, lean=True, device=None, w_t=3, w_d=2, top_n=5):
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



        if self.interest + self.EMB_SUFFIX in self.df.columns:
            embed = self.EMB_SUFFIX
        elif self.interest + self.SENT_EMB_SUFFIX in self.df.columns:
            embed = self.SENT_EMB_SUFFIX
        
        self.df[self.interest + self.CLUSTER_SUFFIX + embed + self.NER_SUFFIX] = None
        unique_clusters = self.df[self.interest + embed + self.CLUSTER_SUFFIX].unique()
        # using tqdm on for i in unique_clusters:
        for i in tqdm(unique_clusters, desc=f"Extracting organizations from {self.interest}"):
            sub = self.df[self.df[self.interest + embed + self.CLUSTER_SUFFIX] == i]
            
            # Adding a column with organizations for each title, body and description
            for col in ['title', 'description', self.interest]:
                sub = sub.assign(**{col+'_orgs': sub[col].apply(lambda x: get_organizations(x) if type(x) == str else [])})

            # getting the n most important organizations for each cluster
            all_orgs = sub['title_orgs'].explode().tolist() * w_t + sub['description'].explode().tolist() * w_d + sub[self.interest + '_orgs'].explode().tolist()
            all_orgs = [org for org in all_orgs if type(org) != float]
            orgs = Counter(all_orgs)
            orgs_list = [org[0] for org in orgs.most_common(top_n)]
            self.df.loc[(self.df[self.interest + embed + self.CLUSTER_SUFFIX] == i), self.interest + self.CLUSTER_SUFFIX + embed + self.NER_SUFFIX] = str(orgs_list)

        return self
    
    def reduce_demensionality(self, n_components=2, n_neighbors=15, min_dist=0.0, metric='cosine'):
        # check if either word_embeddings or sentence embeddings are present else raise error
        embed_column = self._get_embed_col(self.interest)
        
        # getting the embeddings for the column
        embeddings = self.df[embed_column].tolist()

        # creating the umap embeddings
        umap_data = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=metric).fit_transform(embeddings)

        # adding the umap embeddings to the dataframe
        self.df[self.interest + self.REDDIM_SUFFIX + self.EMB_SUFFIX] = umap_data.tolist()

        return self

    def topic_modelling(self, n_words=20):
        '''
        Find topics for each cluster. 

        Parameters
        ----------
        column : str
            The column for which the topics are to be found.
        n_words : int, optional
            The number of words to be displayed for each topic. The default is 20.

        Returns
        -------
        self
            Returns self.
        '''

        # checking if clusters are already created
        cluster_col = self._get_cluster_col(self.interest)

        def lemmatize_text(text):
            # Might consider using more agressive approach aka stemming as well
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            # Lowercase
            text = text.lower()

            # Remove punctuation and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Tokenize and lemmatize
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            return tokens

        # create c-tf-idf helper function
        def c_tf_idf(documents, m, ngram_range=(1, 2)):
            count = CountVectorizer(ngram_range=ngram_range, stop_words="english", tokenizer=lemmatize_text).fit(documents)
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
            return pd.DataFrame(list(top_n_words.items()), columns=[cluster_col, 'topics'])

        # create docs_df with the column and the cluster
        docs_df = self.df[[self.interest, cluster_col]].copy()

        # create docs per topic
        docs_per_topic = docs_df.groupby([cluster_col], as_index = False).agg({self.interest: ' '.join})

        # get tf-idf and count vectorizer
        tf_idf, count = c_tf_idf(docs_per_topic[self.interest].values, len(self.df))

        # get top n words per topic
        top_n_words_df = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n_words)

        # remove cluster_topcs column if already present
        if 'topics' in self.df.columns:
            self.df = self.df.drop('topics', axis=1)

        # merge top n words with self.df
        self.df = self.df.merge(top_n_words_df, on=cluster_col, how='left')
        
        return self
    
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

    def _kmeans_clustering(self, df, embed_column, k_upperbound, method, k, min_k):
        if (k is None) and (method == 'silhouette'):
            sil = []
            K = range(2, k_upperbound)

            # Calculate Silhouette Score for each K
            for k in K:
                kmeans = KMeans(n_clusters=k, n_init='auto').fit(df[embed_column].tolist())
                labels = kmeans.labels_
                sil.append(silhouette_score(df[embed_column].tolist(), labels))

            # Find optimal K
            k = sil.index(max(sil)) + 2  # +2 because index starts from 0 and k starts from 2

        elif (k is None) and (method == 'elbow'):
            # ssd = sum of squared distances
            ssd = []
            K = range(2, k_upperbound)

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
        else:
            raise ValueError("Invalid method. Choose either 'silhouette' or 'elbow'.")

        # k has to be greater or equal to 5
        k = max(k, min_k)

        return KMeans(n_clusters=k, n_init='auto').fit(df[embed_column].tolist())

    def _find_medoids(self, model, n_med, cluster_method):
        df = self.df
        clust_column = self._get_cluster_col(self.interest)
        embed_column = self._get_embed_col(self.interest)

        # create column to indicate medoids
        df[clust_column + self.MEDOID_SUFFIX] = False

        # find medoids according to cluster method
        if cluster_method == 'kmeans':
            centroids = model.cluster_centers_

            for i, centroid in enumerate(centroids):
                points_in_cluster = df[df[clust_column] == i][embed_column]
                distances = points_in_cluster.apply(lambda x: np.linalg.norm(np.array(x) - centroid))
                closest_indices = distances.nsmallest(n_med).index
                df.loc[closest_indices, clust_column + self.MEDOID_SUFFIX] = True

        elif cluster_method == 'hdbscan':
            clusters = df[clust_column].unique()
            for cluster in clusters:
                if cluster == -1:  # HDBSCAN labels noise points with -1
                    continue
                
                points_in_cluster = df[df[clust_column] == cluster][embed_column]
                avg_distances = points_in_cluster.apply(lambda x: points_in_cluster.apply(lambda y: np.linalg.norm(np.array(x) - np.array(y))).mean())
                closest_indices = avg_distances.nsmallest(n_med).index
                df.loc[closest_indices, clust_column + self.MEDOID_SUFFIX] = True