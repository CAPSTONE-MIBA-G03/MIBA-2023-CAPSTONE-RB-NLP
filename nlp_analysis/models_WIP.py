import pandas as pd
import spacy
#!python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_sm")

def extract_top5_orgs(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    orgs = [ent for ent in entities if ent[1] == 'ORG']
    top5_orgs = list(pd.Series([org[0] for org in orgs]).value_counts().head(5).index)
    return top5_orgs

if __name__ == "__main__":
    extract_top5_orgs()