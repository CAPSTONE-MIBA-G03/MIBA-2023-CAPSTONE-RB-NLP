import spacy
from collections import Counter

def extract_entities(df, w1 : int, w2: int, top_n: int):
    """
    Function to extract the top_n most frequently appearing organizations in a text
    :param text: text to extract organizations from
    :param w1: weight for title
    :param w2: weight for description
    :param top_n: number of organizations to return
    :return: list of top_n organizations
    """
    nlp = spacy.load("en_core_web_lg")

    # adding a pattern that will not classify quantum (capital or lower), AI, Company, quantum computing as organizations
    patterns = [{'label': 'NORG', 'pattern': 'Quantum'},
                {'label': 'NORG', 'pattern': 'quantum'},
                {'label': 'NORG', 'pattern': 'AI'},
                {'label': 'NORG', 'pattern': 'Company'},
                {'label': 'NORG', 'pattern': 'quantum computing'},
                {'label': 'NCORG', 'pattern': 'NYSE'}, # not quantum orgs
                {'label': 'NCORG', 'pattern': 'NASDAQ'}]

    # adding the patterns to the nlp model, if it is not already there
    if 'entity_ruler' not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)

    def get_organizations(text):
        doc = nlp(text)
        orgs = [str(ent) for ent in doc.ents if ent.label_ == 'ORG']
        return orgs

    # Adding a column with organizations for each title, body and description
    for col in ['title', 'body', 'se_description']:
        df = df.assign(**{col+'_orgs': df[col].apply(lambda x: get_organizations(x) if type(x) == str else [])})

    all_orgs = df['title_orgs'].explode().tolist() * w1 + df['se_description_orgs'].explode().tolist() * w2 + df['body_orgs'].explode().tolist()
    all_orgs = [org for org in all_orgs if type(org) != float]
    orgs = Counter(all_orgs)
    orgs_list = [org[0] for org in orgs.most_common(top_n)]
    return orgs_list