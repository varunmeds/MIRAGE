import os
import re
import json
import glob
import faiss
# import torch
import numpy as np
import configparser
import pandas as pd

from tqdm import tqdm
from autofaiss import build_index

from sentence_transformers import SentenceTransformer

class AutoFaissSentenceSearch:
    def __init__(self,sentence_model, index_folder, max_index_memory_usage='10MB'):
        self.index_folder = os.path.abspath(index_folder)
        
        if not os.path.exists(self.index_folder):
            os.makedirs(self.index_folder)
            
        self.max_index_memory_usage = max_index_memory_usage
        self.sentence_model = SentenceTransformer(sentence_model)
        self.index = None

    def preprocess_text(self, text):
        # Your existing preprocessing code
        return re.sub(r'[^\w\s,.]', '', text)

    def load_processed_files(self, path):
        if os.path.exists(path):
            with open(path, "r") as file:
                return set(file.read().splitlines())
        return set()

    def save_processed_files(self, processed_files, path):
        with open(path, "w") as file:
            for filename in processed_files:
                file.write(f"{filename}\n")

    def create_dataframe_from_json(self, processed_files_path):
        processed_files = self.load_processed_files(processed_files_path)
        new_data = []
        new_data_found = False
    
        # Directories
        current_working_directory = os.getcwd()
        wiki_dir = os.path.join(current_working_directory,'rag_database','wikipedia')
        bing_dir = os.path.join(current_working_directory,'rag_database','bing_search')
    
        # Process Wikipedia files
        for filename in sorted(os.listdir(wiki_dir)):
            if filename.endswith('.json'):# and filename not in processed_files:
                new_data_found = True
                with open(os.path.join(wiki_dir, filename), 'r') as file:
                    json_data = json.load(file)
                    position = 0  # Reset position counter for each file
                    if 'Title' in json_data:
                        new_data.append({'filename': filename, 'text': json_data['Title'], 'position': position})
                        position += 1
                    if 'Description' in json_data:
                        new_data.append({'filename': filename, 'text': json_data['Description'], 'position': position})
                        position += 1
                    for line in json_data.get('PageText', []):
                        new_data.append({'filename': filename, 'text': line, 'position': position})
                        position += 1
                processed_files.add(filename)
    
        # Process Bing files
        for hash_dir in os.listdir(bing_dir):
            hash_dir_path = os.path.join(bing_dir, hash_dir)
            if os.path.isdir(hash_dir_path):
                for filename in sorted(os.listdir(hash_dir_path)):
                    if filename.endswith('.json'):# and filename not in processed_files:
                        new_data_found = True
                        with open(os.path.join(hash_dir_path, filename), 'r') as file:
                            json_data = json.load(file)
                            position = 0  # Reset position counter for each file
                            if 'Title' in json_data:
                                new_data.append({'filename': filename, 'text': json_data['Title'], 'position': position})
                                position += 1
                            if 'Description' in json_data:
                                new_data.append({'filename': filename, 'text': json_data['Description'], 'position': position})
                                position += 1
                            for line in json_data.get('PageText', []):
                                new_data.append({'filename': filename, 'text': line, 'position': position})
                                position += 1
                        processed_files.add(filename)
    
        if new_data_found:
            new_df = pd.DataFrame(new_data, columns=['filename', 'text', 'position'])
            if hasattr(self, 'df') and self.df is not None:
                self.df = pd.concat([self.df, new_df], ignore_index=True)
            else:
                self.df = new_df
            self.df.to_pickle(os.path.join(self.index_folder, "dataframe.pkl"))
        elif not hasattr(self, 'df') or self.df is None:
            try:
                self.df = pd.read_pickle(os.path.join(self.index_folder, "dataframe.pkl"))
            except FileNotFoundError:
                print("No existing DataFrame found. Please ensure there's at least some initial data.")
        
        self.save_processed_files(processed_files, processed_files_path)
        return self.df
    
    def generate_embeddings(self, dataframe):
        embeddings = []
        for text in tqdm(dataframe['text']):
            preprocessed_text = self.preprocess_text(text)
            embedding = self.sentence_model.encode(preprocessed_text, normalize_embeddings=True)
            embeddings.append(embedding)
        return embeddings

    def save_index(self):
        if self.index is not None:
            self.index.save(self.index_folder)
        else:
            print("No index to save.")

    def load_index(self):
        try:
            print(glob.glob(f"{self.index_folder}/*.index"))
            self.index = faiss.read_index(glob.glob(f"{self.index_folder}/*.index")[0])
            # index = faiss.read_index(glob.glob(f"{args.index_dir}/*.index")[0])
        except FileNotFoundError:
            print("Index file not found.")

    def load_dataframe(self):
        df_path = os.path.join(self.index_folder, "dataframe.pkl")
        if os.path.exists(df_path):
            self.df = pd.read_pickle(df_path)
        else:
            print("DataFrame file not found.")

    def build_index(self):
        processed_files_path = os.path.join(self.index_folder, "processed_files.txt")
        self.df = self.create_dataframe_from_json(processed_files_path)
        
        embeddings = self.generate_embeddings(self.df)
        embeddings_array = np.array(embeddings)

        if not os.path.exists(self.index_folder):
            os.makedirs(self.index_folder)

        # Define the paths for saving the embeddings and index files
        embeddings_path = os.path.join(self.index_folder, "embeddings.npy")
        index_path = os.path.join(self.index_folder, "knn.index")
        index_infos_path = os.path.join(self.index_folder, "infos.json")

        # Save the embeddings to a file
        np.save(embeddings_path, embeddings_array)

        # Build and save the index using AutoFaiss
        build_index(
            embeddings=self.index_folder,
            index_path=index_path,
            index_infos_path=index_infos_path,
            max_index_memory_usage=self.max_index_memory_usage
        )

        self.df.to_pickle(os.path.join(self.index_folder, "dataframe.pkl"))
        
        print(f"Index built and saved to {index_path}")

        
        # embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
        # self.index, _ = build_index(embeddings_tensor.numpy(), save_on_disk=False)

    def search_sentences(self, query, top_k=5, context_size=3):
        preprocessed_query = self.preprocess_text(query)
        q_embedding = self.sentence_model.encode(preprocessed_query, normalize_embeddings=True)
        q_embedding = q_embedding.reshape(1, -1)
        _, I = self.index.search(q_embedding, top_k)

        results = []
        for idx in I[0]:
            file_position = self.df.iloc[idx]['position']
            filename = self.df.iloc[idx]['filename']
    
            # Get context sentences
            start_pos = max(0, file_position - context_size)
            end_pos = min(len(self.df), file_position + context_size + 1)
            context_df = self.df[(self.df['filename'] == filename) & 
                                 (self.df['position'] >= start_pos) & 
                                 (self.df['position'] < end_pos)]
    
            context_before = context_df[context_df['position'] < file_position]['text'].tolist()
            context_after = context_df[context_df['position'] > file_position]['text'].tolist()
            main_sentence = self.df.iloc[idx]['text']
    
            sentence_info = {
                'Main Sentence': main_sentence,
                'Context Before': context_before,
                'Context After': context_after
            }
            results.append(sentence_info)
            if len(results) >= top_k:
                break

        return results

