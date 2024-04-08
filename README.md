# MIRAGE: Multimodal Indexing for Retrieval Augmented Generation Engine

## Overview
MIRAGE is a state-of-the-art retrieval augmented generation system that intelligently leverages data from diverse sources such as Wikipedia and Bing. By employing advanced natural language processing with Spacy/NLTK, MIRAGE deconstructs the gathered information into sentences, which are then encoded using a selected sentence transformer. These sentence embeddings are indexed using autofaiss, forming the backbone of the retrieval system. A standout feature of MIRAGE is its ability to not only retrieve the most relevant sentence embeddings for any given query but also to provide additional context by delivering a specified number of sentences preceding and succeeding the matched sentence, thereby enriching the user's comprehension and the accuracy of responses.

## Setup
Begin by ensuring Python is installed on your system, then install the necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Downloading Data
To initiate the data collection process:

1. Open `src/config.txt` and modify `wiki_terms` and `bing_terms` with the terms you wish to search for.
2. Run the following command within the `src` directory to start downloading data:

    ```bash
    python download_data.py
    ```

   This command will download the data into a directory named `rag_database`.

## Building the Index
After data collection is complete, follow these steps to build the index:

1. Inside the `src` directory, execute:

    ```bash
    python build_index.py
    ```

   This will construct the sentence embeddings index and store it in the `index_folder`, which you can specify in `config.txt`.

## Searching and Retrieving Elements
To conduct searches and retrieve relevant sentences:

1. Insert your query into the `query` section of `config.txt`.
2. From the `src` directory, run:

    ```bash
    python search_with_index.py
    ```

   This operation retrieves the most closely matching sentence embeddings for your query and provides `N` sentences before and after the identified sentence, offering a richer context.

## Cross Encoder Inclusion 
To conduct a re-ranking process for the matched sentence embedding retrieved by the search operation , based on an proximity score for each embedding with the query , via a cross encoder :

1. Open `src/config.txt` and modify `cross_encoder_rerank` to True.
2. 2. From the `src` directory, run:

    ```bash
    python search_with_index.py
    ```

## Future Directions
MIRAGE is continually evolving, with plans to incorporate hypothetical document embeddings and step-back prompting to refine the retrieval process. The long-term vision involves experimenting with multimodal embeddings, potentially integrating with Meta's Imagebind project or the LLAVA project, to explore advanced reasoning and captioning capabilities across various media modalities.

## Contributing
We welcome contributions to MIRAGE, whether it's in the form of feature additions, bug fixes, or documentation enhancements. Feel free to fork the repository, make your changes, and submit a pull request.

## License
MIRAGE is made available under the [MIT License](LICENSE). You are authorized to use, modify, and distribute it subject to the terms of this license.

Your participation and feedback can help make MIRAGE an even more powerful tool for data retrieval and analysis. For any questions, suggestions, or contributions, please don't hesitate to reach out or submit a pull request. Let's advance the capabilities of retrieval augmented generation systems together with MIRAGE!
