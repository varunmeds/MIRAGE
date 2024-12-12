import hashlib
import os
from autofaiss_index import AutoFaissSentenceSearch
import configparser

def md5_hash(sentence):
    return hashlib.md5(sentence.encode()).hexdigest()

def build_and_save_index(sentence_model, index_folder, max_index_memory_usage,matryoshka,matryoshka_dim):
    afss = AutoFaissSentenceSearch(sentence_model=sentence_model, index_folder=index_folder, max_index_memory_usage=max_index_memory_usage,matryoshka=matryoshka,matryoshka_dim=matryoshka_dim)
    
    afss.build_index()
    print("Index built and saved.")

if __name__ == "__main__":

    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.txt')
    
    model = config['DEFAULT']['model']
    index_folder = config['DEFAULT']['index_folder']
    max_index_memory_usage = config['DEFAULT'].get('max_index_memory_usage', '10MB')
    matryoshka_flag=config['DEFAULT']['matryoshka']
    matryoshka_dim=config['DEFAULT']['matryoshka_dim']

    build_and_save_index(model, index_folder, max_index_memory_usage,matryoshka_flag,matryoshka_dim)
