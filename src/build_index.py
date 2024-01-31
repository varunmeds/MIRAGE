import hashlib
import os
from autofaiss_index import AutoFaissSentenceSearch
import configparser

def md5_hash(sentence):
    return hashlib.md5(sentence.encode()).hexdigest()

def build_and_save_index(sentence_model, index_folder, max_index_memory_usage):
    afss = AutoFaissSentenceSearch(sentence_model=sentence_model, index_folder=index_folder, max_index_memory_usage=max_index_memory_usage)
    
    afss.build_index()
    print("Index built and saved.")

if __name__ == "__main__":

    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.txt')
    
    model = config['DEFAULT']['model']
    index_folder = config['DEFAULT']['index_folder']
    max_index_memory_usage = config['DEFAULT'].get('max_index_memory_usage', '10MB')

    build_and_save_index(model, index_folder, max_index_memory_usage)