from autofaiss_index import AutoFaissSentenceSearch
from CrossEncoderSearch import CrossencoderSearch, dict_to_list
import configparser
import time
import os
import json
import random

def load_index_and_search(query, sentence_model,index_folder,max_index_memory_usage,matryoshka,matryoshka_dim):
    afss = AutoFaissSentenceSearch(sentence_model=sentence_model, index_folder=index_folder, max_index_memory_usage=max_index_memory_usage,matryoshka=matryoshka,matryoshka_dim=matryoshka_dim)
    afss.load_index()
    afss.load_dataframe() 
    search_results = afss.search_sentences_test(query)
    return search_results

def load_index_and_test(query, sentence_model, index_folder, max_index_memory_usage,matryoshka,matryoshka_dim):
    afss = AutoFaissSentenceSearch(sentence_model=sentence_model,index_folder=index_folder,max_index_memory_usage=max_index_memory_usage,matryoshka=matryoshka,matryoshka_dim=matryoshka_dim)
    current_working_directory = os.getcwd()
    test_dir = os.path.join(current_working_directory,'rag_database')
    test_file_path = os.path.join(test_dir,query)
    with open(test_file_path, 'r') as file:
        loaded_text_list = json.load(file)
    #random.shuffle(loaded_text_list)
    '''split_index = int(len(loaded_text_list) * 0.2853)
    texts_query = loaded_text_list[:split_index]
    #texts_list = texts[split_index:]
    texts_query_split_index = int(len(texts_query) * 0.2)
    texts_query_small = texts_query[:texts_query_split_index]
    texts_list_small = texts_query[texts_query_split_index:]'''
    subset_list = loaded_text_list[:20000]
    texts_list_query = subset_list[:2000]
    texts_list_database = subset_list[2000:20000]
    print("Start of function")
    df = afss.create_dataframe_from_texts(texts_list_database)
    print("database created")
    afss.save_dataframe_from_texts(df, filename="test_dataframe.pkl")
    print("database saved")
    afss.build_index_from_texts(df,8,2000,20000)
    print("index built for database")
    afss.load_index()
    print("index loaded")
    search_results = afss.search_sentences_test(texts_list_query)
    print("End of function")
    return search_results

if __name__ == "__main__":

    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.txt')
    
    query = config['DEFAULT']['query']
    #query = config['DEFAULT']['query'].split(',')
    test_flag = config['DEFAULT']['test_flag']
    model = config['DEFAULT']['model']
    index_folder = config['DEFAULT']['index_folder']
    max_index_memory_usage = config['DEFAULT'].get('max_index_memory_usage', '10MB')
    matryoshka = config.getboolean('DEFAULT', 'matryoshka')
    matryoshka_dim = config['DEFAULT']['matryoshka_dim']

    #the query field in the config must change depending on the text_flag, for true it must be a file path , for false it must be a singular query 
    if(test_flag=="True"):
        results = load_index_and_test(query, model, index_folder, max_index_memory_usage,matryoshka,matryoshka_dim)
    else:
        results = load_index_and_search(query, model, index_folder, max_index_memory_usage,matryoshka,matryoshka_dim)
    #input=dict_to_list(results)
    print(results)
    '''file_name = 'query_times_64.txt'
    with open(file_name, 'w') as file:
        json.dump(results, file)'''

    '''if(config['DEFAULT']['cross_encoder_rerank']):
        ce = CrossencoderSearch(query,input)
        outputs = ce.run_cross_encoder()
        print(outputs)
    else:
        print(results)'''
