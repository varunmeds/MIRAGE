from autofaiss_index import AutoFaissSentenceSearch
from CrossEncoderSearch import CrossencoderSearch, dict_to_list
import configparser

def load_index_and_search(query, sentence_model,index_folder,max_index_memory_usage):
    afss = AutoFaissSentenceSearch(sentence_model=sentence_model, index_folder=index_folder, max_index_memory_usage=max_index_memory_usage)
    afss.load_index()
    afss.load_dataframe() 
    search_results = afss.search_sentences(query)
    return search_results

if __name__ == "__main__":

    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.txt')
    
    query = config['DEFAULT']['query']
    model = config['DEFAULT']['model']
    index_folder = config['DEFAULT']['index_folder']
    max_index_memory_usage = config['DEFAULT'].get('max_index_memory_usage', '10MB')
    
    results = load_index_and_search(query, model, index_folder, max_index_memory_usage)
    input=dict_to_list(results)

    if(config['DEFAULT']['cross_encoder_rerank']):
        ce = CrossencoderSearch(query,input)
        outputs = ce.run_cross_encoder()
        print(outputs)
    else:
        print(results)
