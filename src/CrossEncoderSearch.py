from sentence_transformers import CrossEncoder

class CrossencoderSearch:

    def __init__(self,query,retrieved_docs):

        self.query=query
        self.pairs=retrieved_docs
        self.final_pairs=[]
        self.cross_encoding_model= CrossEncoder('BAAI/bge-base-en-v1.5')

    def run_cross_encoder(self):
        self.final_pairs = [[self.query, doc_text, self.cross_encoding_model.predict([self.query, doc_text])] for doc_text in self.pairs]
        self.final_pairs.sort(key=lambda x: x[2])
        self.final_pairs.reverse()
        return self.final_pairs
