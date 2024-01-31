import os
import json
import hashlib
import wikipediaapi

def md5_hash(sentence):
    return hashlib.md5(sentence.encode()).hexdigest()

class WikipediaPageFetcher:
    def __init__(self, user_agent, language='en', extract_format=wikipediaapi.ExtractFormat.WIKI):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=language,
            extract_format=extract_format
        )

    def get_wikipedia_page_content(self, page_title):
        """
        Fetches a Wikipedia page content.

        Args:
        page_title (str): Title of the Wikipedia page to fetch.

        Returns:
        str: Content of the Wikipedia page.
        """
        page = self.wiki_wiki.page(page_title)
        if not page.exists():
            return None
        return page

    def save_content_to_file(self, page, base_folder='wikipedia'):
        if not page:
            return "Page not found"

        page_title = page.title
        content = page.text
        hash_value = md5_hash(page_title)
        file_name = f"{hash_value}.json"
        folder_path = os.path.join(os.getcwd(), 'rag_database', base_folder)
        file_path = os.path.join(folder_path, file_name)

        os.makedirs(folder_path, exist_ok=True)

        # Extracting a short description from the content
        description = ' '.join(content.split('\n')[:2]) if content else ''

        page_data = {
            "Title": page_title,
            "Link": page.fullurl,
            "Description": description,
            "Position": "",
            "PageText": content.split("\n"),
            "Hash": hash_value
        }

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(page_data, file, indent=4)

        return f"Content saved to {file_path}"
    
    def fetch_pages_from_file(self, file_path, base_folder='wikipedia'):
        """
        Reads a list of page titles from a file and downloads each page if not already downloaded.

        Args:
        file_path (str): Path to the file containing page titles.
        base_folder (str): Base folder where the content will be saved.

        Returns:
        None
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            page_titles = file.readlines()

        for title in page_titles:
            title = title.strip()
            if title:
                # Sanitize file name and check if it already exists
                sanitized_title = title.replace(" ", "_").replace("/", "_") + '.txt'
                
                current_working_directory = os.getcwd()
                folder_path = os.path.join(current_working_directory,'rag_database', base_folder)
                file_path = os.path.join(folder_path, sanitized_title)
                
                if os.path.exists(file_path):
                    print(f"File already exists for page: {title}")
                    continue

                content = self.get_wikipedia_page_content(title)
                if content != "Page not found":
                    self.save_content_to_file(content, title, base_folder)
                    print(f"Downloaded and saved page: {title}")
                else:
                    print(f"Page not found: {title}")
                    
                    
    # def upload_files_to_s3(self, folder_path, bucket_name, s3_folder):
    #     """
    #     Uploads files from a local directory to an S3 bucket.

    #     Args:
    #     folder_path (str): Path to the local folder containing files to upload.
    #     bucket_name (str): Name of the S3 bucket.
    #     s3_folder (str): S3 folder name (prefix) where files will be stored.

    #     Returns:
    #     None
    #     """
    #     s3_client = boto3.client('s3')
    #     for filename in os.listdir(folder_path):
    #         local_path = os.path.join(folder_path, filename)
    #         s3_path = os.path.join(s3_folder, filename)
    #         try:
    #             s3_client.upload_file(local_path, bucket_name, s3_path)
    #             print(f"Uploaded {filename} to S3 bucket {bucket_name}")
    #         except NoCredentialsError:
    #             print("Credentials not available")
    #             return