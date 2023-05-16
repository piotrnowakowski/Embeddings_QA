import pinecone
import pandas as pd
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
from embedding_creation_whole import create_embeddings

file_path = 'documentation/processed_embeddings_whole.json'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        data_json = json.load(f)
    # Proceed with further operations on the loaded data
else:
    # Handle the case when the file doesn't exist
    print("File 'parsed_data.json' does not exist. I will create it for You")
    create_embeddings()
    try:
        with open(file_path, 'r') as f:
            data_json = json.load(f)
    except:
        print("There was an error creating json data")
        exit()

# loading api keys
with open('openai_key.txt', 'r') as file:
    openai_key = file.read().strip()
with open('pinecone_key.txt', 'r') as file:
    pinecone_key = file.read().strip()

df = pd.DataFrame(data_json)
 
# connect to pinecone environment
pinecone.init(
    api_key=pinecone_key,
    environment="asia-southeast1-gcp-free"  
)

index_name = "telnyx-qa"
# check if the pinecone index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=384,
        metric="cosine"
    )

index = pinecone.Index(index_name)

# set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the retriever model from huggingface model hub
retriever = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)

from tqdm.auto import tqdm

# we will use batches of 64
batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    # extract batch
    batch = df.iloc[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch['combined_text'].tolist()).tolist()
    # get metadata
    meta = batch.to_dict(orient='records')
    # create unique IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

# check that we have all vectors in index
index.describe_index_stats()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = 'deepset/electra-base-squad2'
# load the reader model into a question-answering pipeline
reader = pipeline(tokenizer=model_name, model=model_name, task='question-answering', device=device)


def get_context(question, top_k):
    # generate embeddings for the question
    xq = retriever.encode([question]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    # extract the context passage from pinecone search result
    c = [x["metadata"]['combined_text'] for x in xc["matches"]]
    return c
def extract_answer(question, context):
    results = []
    for c in context:
        # feed the reader the question and contexts to extract answers
        answer = reader(question=question, context=c)
        # add the context to answer dict for printing both together
        answer["combined_text"] = c
        results.append(answer)
    # sort the result based on the score from reader model
    sorted_result = print(sorted(results, key=lambda x: x['score'], reverse=True))
    return sorted_result


question = "What country uses the country dial code 52"
context = get_context(question, top_k = 3)
print(context)
answer = extract_answer(question, context)
print(answer)
