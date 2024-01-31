import requests
import hashlib
import os
import json
import time

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def md5_hash(sentence):
    return hashlib.md5(sentence.encode()).hexdigest()

def save_json_to_file(data, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, f'{filename}.json'), 'w') as file:
        json.dump(data, file, indent=4)

def download_text_from_link(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator='|', strip=True)
        return text
    else:
        return "Failed to retrieve content"

def save_search_query_hash(query, hash, directory):
    log_file_path = os.path.join(directory, 'search_queries_hashes.txt')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the query has already been logged
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            if any(query in line for line in file):
                return

    with open(log_file_path, 'a') as file:
        file.write(f"{query}: {hash}\n")

def fetch_bing_search_results(query, num_pages=10, results_per_page=10,delay=30):

    current_working_directory = os.getcwd()
    base_directory = os.path.join(current_working_directory, 'rag_database', 'bing_search')
    log_directory = os.path.join(current_working_directory, 'rag_database', 'bing_search_logs')

    search_results = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    encoded_query = quote_plus(query)

    # Generating first hash
    first_hash = md5_hash(query)
    directory_name = os.path.join(base_directory, first_hash)

    # Save search query and hash
    save_search_query_hash(query, first_hash, log_directory)

    for page in range(0, num_pages * results_per_page, results_per_page):
        target_url = f"https://www.bing.com/search?q={encoded_query}&rdr=1&first={page+1}"
        
        try:
            response = requests.get(target_url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching page: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            complete_data = soup.find_all("li", {"class": "b_algo"})

            if not complete_data:
                print("No data found on the page.")
                continue

            for position, item in tqdm(enumerate(complete_data, start=1)):
                link = item.find("a").get("href")
                second_hash = md5_hash(link)

                # Check if this link's content has already been downloaded
                json_file_path = os.path.join(directory_name, f'{second_hash}.json')
                if not os.path.exists(json_file_path):
                    page_text = download_text_from_link(link)
                    page_text = [i for i in page_text.split("|") if len(i.split(" "))>7]

                    # Check if page text was successfully retrieved
    #                 if page_text != "Failed to retrieve content":
                    result = {
                        "Title": item.find("a").text,
                        "Link": link,
                        "Description": item.find("div", {"class": "b_caption"}).text,
                        "Position": position,
                        "PageText": page_text,
                        "Hash": second_hash
                    }
                    search_results.append(result)
                    # Save each successful result as a separate JSON file
                    save_json_to_file(result, directory_name, second_hash)
        
        except RequestException as e:
            print(f"Request failed: {e}")
            continue
        
        # Introduce a delay between requests
        time.sleep(delay)
        print("Sleep for 30 seconds")
    
    return response.text,search_results