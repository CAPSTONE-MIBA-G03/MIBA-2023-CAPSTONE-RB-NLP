from transformers import logging
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
import torch
from nltk.tokenize import sent_tokenize

logging.set_verbosity_error()

def bert_embed_words(text):
    '''
    Takes in a string and returns an embedding calculated as the average of embeddings
    per sentence.
    '''
    # Setup device agnostic code (Chooses NVIDIA or Metal backend if available, otherwise defaults to CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        
    else:
        device = torch.device("cpu")

    
    sentences = sent_tokenize(text)

    # make sure there are no empty texts passed in

    # SBERT automatically takes care of reducing input length if too long

    embeddings = model.encode(sentences)

    embeddings = embeddings.mean(axis=0)

    return embeddings