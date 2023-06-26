# MIBA-2023-CAPSTONE-RB-NLP (Esade Project)
This project from ESADE MIBA students of team G03 in collaboration with Roland Berger Paris aims to develop a data ingestion pipeline for news article data and test different NLP approaches on the extracted data to analyze topics and trends linked to a particular brand or company name.

ETL Pipeline Components:
- Link Extractor
- Content Extractor
- Content Cleaner

NLP Pipeline Components:
- WordWizard

# Guide to Using PipelineExecutor and WordWizard Modules

## Introduction

The `PipelineExecutor` and `WordWizard` are robust modules built to extract, clean, analyze, and visualize data pertaining to any given topic. They offer a robust solution to perform complex Natural Language Processing (NLP) tasks efficiently. This README aims to provide comprehensive instructions on how to utilize these modules.

## Installation

To begin, make sure these modules are correctly installed and imported in your Python environment:

```python
from etl_pipeline import PipelineExecutor
from nlp_analysis import WordWizard
```

## Using the PipelineExecutor Module

`PipelineExecutor` is a powerful module designed to extract articles from the web on a specific topic, clean the data for further analysis, and cache the data to avoid repeated extractions for the same topic.

### Initialization

```python
# Instantiate the PipelineExecutor
pipeline = PipelineExecutor()
```

### Extracting and Cleaning Content

```python
# Define the topic
topic = 'Quantum Computing'

# Execute the pipeline for the topic
df = pipeline.execute(topic)
```

The `execute` method carries out the following steps:

1. **Extracts links**: It fetches links related to the topic from various search engines.
2. **Extracts content**: It then visits each link and extracts the relevant content.
3. **Cleans the content**: The extracted content undergoes cleaning, which involves removing irrelevant information, special characters, extra white spaces, etc.
4. **Caches the content**: The cleaned data is cached locally to prevent repeated extraction for the same topic.

The method returns a pandas DataFrame containing the clean content.

## Using the WordWizard Module

`WordWizard` is an extensive module supporting various NLP tasks like creating word embeddings, clustering, text summarization, sentiment analysis, entity recognition, dimensionality reduction, and topic modeling.

### Initialization

```python
# Instantiate the WordWizard with the DataFrame from PipelineExecutor
wizard = WordWizard(df)
```

### Processing and Analyzing the Content

The processing and analysis of the content can be performed by chaining the methods as follows:

```python
wizard.create_word_embeddings() \
    .cluster_embeddings()\
    .summarize_medoids()\
    .find_sentiment()\
    .entitiy_recognition()\
    .reduce_demensionality()\
    .topic_modelling()
```

Each of these methods updates the DataFrame associated with the `WordWizard` object and performs a specific operation:

1. **`create_word_embeddings`:** This method creates embeddings (dense vector representations) for the words in the content.
2. **`cluster_embeddings`:** This method groups the embeddings into clusters using a specified algorithm.
3. **`summarize_medoids`:** It generates a summary of the clusters by identifying the medoids (the most representative points of a cluster).
4. **`find_sentiment`:** This method computes the sentiment score for the text content.
5. **`entitiy_recognition`:** This method identifies and extracts the entities present in the text.
6. **`reduce_demensionality`:** It reduces the dimensionality of the data for better visualization and understanding.
7. **`topic_modelling`:** It identifies the main topics that emerge from the text content.

## Analyzing the DataFrame Within WordWizard

After executing the chained processing and analysis operations on the WordWizard object, you can access the updated DataFrame for further analysis. This DataFrame incorporates the results of all the NLP tasks performed by the WordWizard.

```python
# Access the DataFrame
analyzed_df = wizard.df
```

## Conclusion

With `PipelineExecutor` and `WordWizard` modules, you can build a comprehensive pipeline to perform complex NLP tasks. These modules offer an easy-to-use and efficient way to extract, clean, analyze, and visualize data from the web.