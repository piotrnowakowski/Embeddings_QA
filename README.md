# Documentation for SMS Guidelines Retrieval System

This system provides answers to user queries in the context of "SMS Guidelines for countries with additional info". It uses natural language understanding and machine learning techniques to find the most appropriate answer from a corpus of documents.

## Purpose of the Application

The application is designed to provide users with answers to their questions using data embedded in a set of documents about SMS Guidelines for various countries. The application uses OpenAI's GPT, to understand the context and semantics of the queries and the documents in the corpus and embeddings stored in Pinecone database to do search for relevant context.

## Libraries Used

-   langchain: A Python library designed to enable efficient use of large language models and vector database.
-   pinecone: A vector database that provides fast and scalable operations for embedding vectors.
-   pandas: A data analysis and manipulation tool.
-   numpy: A library for numerical operations.
-   json: A library for parsing and handling JSON data.
-   os: A module for interacting with the operating system.
-   BeautifulSoup: A library for parsing HTML and XML documents.

## How to Run the Application

1.  Ensure that you have installed all the necessary libraries. If not, install them using pip.


    pip install langchain pinecone pandas numpy beautifulsoup4 

2.  Clone the repository to your local machine.

    git clone <repository_url> 

3.  Navigate into the cloned repository.

    `cd <repository_name>` 

4.  Before running the application, you need to have the API keys for OpenAI and Pinecone. Save them in two separate text files named `openai_key.txt` and `pinecone_key.txt` respectively.
    
5.  Run the main application file, `main.py`.
    



    `python main.py` 

6.  The application will now prompt you to ask a question. Input your question and press Enter. The application will return the most relevant answer from the corpus of documents.

## Files in the Application

1.  `main.py`: This is the main application file that orchestrates the process of loading the data, embedding it, querying, and returning the answer.
    
2.  `embedding_creation_whole.py`: This script is responsible for cleaning the raw HTML content, extracting the necessary information, and generating the document embeddings.
    
3.  `openai_key.txt` and `pinecone_key.txt`: These text files contain the API keys for OpenAI and Pinecone.
    
4.  `parsed_data.json`: This file contains the preprocessed and cleaned data which is loaded by the main application.
    

## Data Used

The application uses data from the file `parsed_data.json`. This file contains cleaned and processed data about SMS Guidelines for various countries.

## Architecture of the Application

The application follows a series of steps to deliver the answers:

1.  It loads the data from the `parsed_data.json` file. If the file doesn't exist, it generates the embeddings using the `create_embeddings()` function from `embedding_creation_whole.py`.
    
2.  It initializes the Pinecone environment with the provided API key and establishes a connection.
    
3.  It prepares the data by handling any NaN values and loads it into a DataFrame.
    
4.  It uses the OpenAIEmbeddings class to generate embeddings for the documents.
    
5.  It checks if the Pinecone index exists. If not, it creates the index and loads the embeddings into Pinecone.
    
6.  It sets up a SelfQueryRetriever to use the metadata provided in the embeddings.
    
7.  It sets up a ConversationBufferMemory to store the chat history.
    
8.  It initializes a ConversationalRetrievalChain with the OpenAI language model, the retriever, and the memory.
    
9.  The application now enters into a loop, where it accepts user queries, processes them, and returns the most appropriate  answers from the corpus.