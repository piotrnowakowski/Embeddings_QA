#langchain imports
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DataFrameLoader
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
# external lib import
import pinecone
import os
import pandas as pd
import numpy as np
import json
# internal functions import
from embedding_creation_whole import create_embeddings


# loading api keys
with open('openai_key.txt', 'r') as file:
    openai_key = file.read().strip()
with open('pinecone_key.txt', 'r') as file:
    pinecone_key = file.read().strip()

 
# connect to pinecone environment
pinecone.init(
    api_key=pinecone_key,
    environment="asia-southeast1-gcp-free"  
)

#loading or generating file
file_path = 'documentation/parsed_data.json'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_json(f, encoding='utf-8')

else:
    # Handle the case when the file doesn't exist
    print("File 'parsed_data.json' does not exist. I will create it for You")
    create_embeddings()
    try:
        with open(file_path, 'r') as f:
            df = pd.read_json(f, encoding='utf-8')

    except:
        print("There was an error creating json data")
        exit()

# handling nan values if there are some
df = df.replace(np.nan, '')

loader = DataFrameLoader(df, page_content_column='combined_text')

embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Defining metada to enhance the embeddings search
metadata_field_info=[
    AttributeInfo(
        name="title",
        description="Name of the country", 
        type="string", 
    ),
    AttributeInfo(
        name="MCC",
        description="MCC stands for Country Code - three-digit code used in SMS communication with a country name", 
        type="int", 
    ),
    AttributeInfo(
        name="Dial Code",
        description="A dial code in mobile phone communications is a series of digits used to identify a specific geographic location when making a phone call.", 
        type="int", 
    ),
    AttributeInfo(
        name="Alphanumeric",
        description="Alphanumeric refers to a combination of letters and numbers used in mobile communication for creating and sending text messages.",
        type="string"
    ),
]
page_content = "SMS Guidelines for country with additional info"
langchain_document = loader.load()


index_name = "telnyx-qa"
# check if the pinecone index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric="cosine"
    )
    vectorstore = Pinecone.from_documents(langchain_document, embeddings, index_name=index_name)
else:
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

#set up selfqueryretriever to use the metadata provided in the embeddings
llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, page_content, metadata_field_info, verbose=True)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
i=0
chat_history = []
print("You can start asking question")
while i<5:
    i += 1
    query = input()
    result = qa({"question": query})
    retriever_response = retriever.get_relevant_documents(query)[0].page_content
    print(retriever_response)
    chat_history.append((query, result["answer"]))
    for j in chat_history:
        print("user: {} \n bot:{}".format(j[0], j[1]))
