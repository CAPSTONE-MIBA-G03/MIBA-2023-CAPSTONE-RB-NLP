import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from tqdm.auto import tqdm

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

    def __init__(self, df, device=None) -> None:
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

        self.df["sentences"] = self.df["paragraph"].apply(lambda x: sent_tokenize(x))

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

        new_column_name = column + self.EMB_SUFFIX
        self.df[new_column_name] = np.nan

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
        self.df[column + self.SENT_EMB_SUFFIX] = self.df[column].progress_apply(lambda sentences: model.encode(sentences, device=device).mean(axis=0))

        return self

    def cluster_embeddings(self, column, k_upperbound=15, extra_clusters=1, method=None, k=5, n_med=2):
        if column == 'body':
            # getting unique articles
            df = self.df[~self.df['article_index'].duplicated()]
        else:
            df = self.df

        # Define main variables
        kmeans = {}
        K = range(2, k_upperbound)
        if column == 'sentences':
            column = column + self.SENT_EMB_SUFFIX
        else:
            column = column + self.EMB_SUFFIX # e.g.: paragraph_word_embeddings

        # Determine optimal K
        if method == None:
            optimal_k = k

        elif method == 'silhouette':
            sil = []

            # Calculate Silhouette Score for each K
            for k in K:
                kmeans[k] = KMeans(n_clusters=k, n_init='auto').fit(df[column].tolist())
                labels = kmeans[k].labels_
                sil.append(silhouette_score(df[column].tolist(), labels, metric='euclidean'))

            # Find optimal K
            optimal_k = sil.index(max(sil)) + 2  # +2 because index starts from 0 and k starts from 2
        
        elif method == 'elbow':
            # ssd = sum of squared distances
            ssd = []

            # Calculate sum of squared distances for each K
            for k in K:
                model = KMeans(n_clusters=k, n_init='auto').fit(df[column].tolist())
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

        kmeans = KMeans(n_clusters=optimal_k, n_init='auto').fit(df[column].tolist())
        new_column = column + self.CLUSTER_SUFFIX # e.g.: paragraph_word_embeddings_cluster_5
        
        # Add cluster labels to dataframe
        df[new_column] = kmeans.labels_
        
        # Finding Medoids
        centroids = kmeans.cluster_centers_
        df[new_column + self.MEDOID_SUFFIX] = False
        for i, centroid in enumerate(centroids):
            points_in_cluster = df[df[new_column] == i][column]
            distances = points_in_cluster.apply(lambda x: np.linalg.norm(np.array(x) - centroid))
            closest_indices = distances.nsmallest(n_med).index
            df.loc[closest_indices, new_column + self.MEDOID_SUFFIX] = True
            
        # filling the initial df (self.df)
        for i, pos in enumerate(df.index):
            # if the current index is the last index in df, fill the rest of the df
            if i + 1 == len(df.index):
                self.df.loc[pos:, new_column] = df.loc[pos:, new_column]
                self.df.loc[pos:, new_column + self.MEDOID_SUFFIX] = df.loc[pos:, new_column + self.MEDOID_SUFFIX]
            # else, fill the df until the next index
            else:
                next_index = df.index[i + 1]
                self.df.loc[pos:next_index - 1, new_column] = df.loc[pos:next_index - 1, new_column]
                self.df.loc[pos:next_index - 1, new_column + self.MEDOID_SUFFIX] = df.loc[pos:next_index - 1, new_column + self.MEDOID_SUFFIX]
        
        #self.df = self.df.merge(df[['article_index', new_column, new_column + self.MEDOID_SUFFIX]], on='article_index', how='left')    
        return self

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


        summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        medoids = self.df[self.df[column + '_word_embeddings_clusters' + self.MEDOID_SUFFIX] == True] # body_word_embeddings_clusters_medoids
        self.df[column + self.SUMMARY_SUFFIX] = None
        for pos in medoids.index:
            entry = medoids.loc[pos, column]
            if type(entry) != str:
                summary = 'summarized entry was no string'
            tokenized_input = tokenizer.tokenize(entry)
            if len(tokenized_input) > 1023: # max input size for BART
                entry = ' '.join(tokenized_input[:1000]).replace(' ##', '')
            try:
                summary = summarizer(entry) # set max size (! >30)

            except:
                summary = 'could not summarize'

            if type(summary) == list:
                summary = summary[0]['summary_text']

            if type(summary) == dict:
                summary = summary['summary_text']

            self.df.loc[pos, column + self.SUMMARY_SUFFIX] = summary
            

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
    
    
    def entitiy_recognition(self, columns: list([str]), lean=True, device=None):
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

    def topic_modelling(self, column, with_embeddings=True, sample_size=1000):
        """
        Performs topic modelling on the input column and adds a new column with the suffix '_topic.'

        Parameters
        ----------
        column : {"title", "description", "body", "paragraph", "sentences"}
            The column to perform topic modelling on.

        with_embeddings : bool, optional
            Whether to use the embeddings or not. If not specified, the embeddings are used.

        return_model : bool, optional 
            Whether to return the topic model or not. If not specified, the topic model is not returned.

        Examples
        --------
        >>> from nlp_analysis import WordWizard
        >>> pipe = WordWizard(df = df)
        >>> pipe.topic_modelling(column = "body")
        >>> pipe.df["body_topic"].head()
        
        """

        if with_embeddings:
            # Assume embeddings column is the column name plus "_word_embeddings"
            emb_column = column + self.EMB_SUFFIX
            
            # Check if embeddings exist
            if emb_column not in self.df.columns:
                raise ValueError(f"Embeddings for column {column} not found. Please run 'create_word_embeddings' first.")
            
            # Get embeddings
            embeddings = np.array(self.df[emb_column].tolist())
            
            # Run BERTopic
            self.topic_model = BERTopic(verbose=True)
            topics, _ = self.topic_model.fit_transform(self.df[column], embeddings)

        else:
            # Run BERTopic
            representation_model = KeyBERTInspired()
            self.topic_model = BERTopic(representation_model=representation_model)
            df_sample = self.df.sample(sample_size) # Sample dataframe to reduce computation time
            topics, _ = self.topic_model.fit_transform(df_sample[column])

        # Create a mapping of topic id to words
        topic_id_to_words = {topic_id: self.topic_model.get_topic(topic_id) for topic_id in set(topics)}

        # Add topic labels and words to dataframe
        self.df[column + '_topic_id'] = topics
        self.df[column + '_topic_words'] = self.df[column + '_topic_id'].map(topic_id_to_words)
        
        return self
