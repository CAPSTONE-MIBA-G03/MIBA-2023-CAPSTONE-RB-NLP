import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import logging
logging.set_verbosity_error()


def insert_sep_token(text):
    sentences = sent_tokenize(text)
    text_with_sep = ' [SEP] '.join(sentences)
    return text_with_sep


def bert_embed_text(text):
    # if the input is not a string, return an empty list
    if type(text) != str:
        return []
    marked_text = "[CLS] " + insert_sep_token(text)
    tokenized_text = tokenizer.tokenize(marked_text)

    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]

    # Map the token strings to their vocabulary indeces
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create segment ids (alternating between 0 and 1)
    segments_ids = []
    current_segment_id = 0
    for value in tokenized_text:
        segments_ids.append(current_segment_id)
        if value == "[SEP]":
            current_segment_id = 1 - current_segment_id

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True) # output all hidden states
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # hidden states from all layers because we set output_hidden_states = True
        # See the documentation for more details: # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]


    token_vecs = hidden_states[-1][0] # second to last layer
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding