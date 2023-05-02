import torch
from transformers import BertTokenizer, BertModel
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text1 = "Electric vehicles are becoming increasingly popular. They help reduce greenhouse gas emissions and air pollution. Many governments offer incentives to promote the adoption of electric cars. Charging infrastructure is rapidly expanding in urban areas. The future of transportation seems to be electric."
text2 = "Renewable energy sources are gaining traction worldwide. Solar and wind power are becoming more cost-effective and efficient. Governments are implementing policies to encourage the use of clean energy. Innovations in energy storage, such as advanced batteries, facilitate the adoption of renewables. The shift towards sustainable energy is gaining momentum."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def insert_sep_token(text):
    sentences = sent_tokenize(text)
    text_with_sep = ' [SEP] '.join(sentences)
    return text_with_sep

def bert_embed_text(text):
    marked_text = "[CLS] " + insert_sep_token(text)
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create segment ids
    segments_ids = []
    current_segment_id = 0
    for value in tokenized_text:
        segments_ids.append(current_segment_id)
        if value == "[SEP]":
            current_segment_id = 1 - current_segment_id

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    
    # AVeraging second last layer
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding
