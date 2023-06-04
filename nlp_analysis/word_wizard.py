# Hogwards

class WordWizard:

    def __init__(self, df) -> None:
        self.df = df

    def create_embeddings(self, column):
        pass
        
    def cluster_embeddings(self, column):
        pass

    def find_medoids(self, column):
        pass

    def find_sentiment(self, column):
        pass

    def entitiy_recognition(self, column):
        pass



if __name__ == "__main__":
    pipe = WordWizard(df = df)
