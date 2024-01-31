from wikipedia_fetcher import WikipediaPageFetcher
from bing_search import fetch_bing_search_results
import configparser

def download_wikipedia_content(wiki_search_terms):
    wiki_fetcher = WikipediaPageFetcher(user_agent='MyProjectName (merlin@example.com)')
    for wiki_search_term in wiki_search_terms.split(','):
        wiki_page = wiki_fetcher.get_wikipedia_page_content(wiki_search_term.strip())
        if wiki_page:
            print(wiki_fetcher.save_content_to_file(wiki_page))
        else:
            print("Wikipedia page not found for:", wiki_search_term)

def download_bing_content(bing_search_terms):
    for bing_search_term in bing_search_terms.split(','):
        print(f"Downloading Bing search results for: {bing_search_term}")
        fetch_bing_search_results(bing_search_term.strip())

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.txt')
    
    wiki_terms = config['DEFAULT']['wiki_terms']
    bing_terms = config['DEFAULT']['bing_terms']

    download_wikipedia_content(wiki_terms)
    download_bing_content(bing_terms)