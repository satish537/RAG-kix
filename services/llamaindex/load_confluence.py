from llama_index.readers.confluence import ConfluenceReader
import os, chromadb, shutil
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from services.llamaindex.embedding import get_embedding_function
from services.langchain.text_to_query import text_to_query
from llama_index.llms.ollama import Ollama
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *

path = "././chroma/confluence"

def get_confluence_data_as_vector_llamaindex(url, llm_model, query, username, password, space_key):
    
    collection_name = sanitize_collection_name(space_key)
    COLLECTION_PATH = f"{path}/{collection_name}"

    all_entries = os.listdir(path)
    confluence_directories = [entry for entry in all_entries if os.path.isdir(os.path.join(path, entry))]

    if collection_name in confluence_directories:
        print("Data Available in Chroma Database")

        Settings.llm = Ollama(model=llm_model, request_timeout=1500.0)
        db = chromadb.PersistentClient(path=f"{COLLECTION_PATH}/{collection_name}")

        chroma_collection = db.get_collection(f"{collection_name}")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, embed_model=get_embedding_function()
        )

        file_path = f"{COLLECTION_PATH}/{collection_name}.txt"

        # Read the content of the file
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                file_content = file.read()
        
        return index, file_content



    docs = []

    Settings.llm = Ollama(model=llm_model, request_timeout=1500.0)
    Settings.chunk_size = 2000
    Settings.chunk_overlap = 500

    os.environ["CONFLUENCE_USERNAME"] = username
    os.environ["CONFLUENCE_PASSWORD"] = password
    base_url = url
    space_key = space_key

    words = query.split()

    # Construct the CQL query by combining each word with the OR operator
    title_query = ' OR '.join([f'text ~ "{word}"' for word in words])
    # Add the space ID filter to the query
    cql_query = f'space = "{space_key}" AND ({title_query})'




    # cql = f'space = "{space_key}" AND text ~ "{query}"'
    # cql = f'text ~ "Software Testing"'

    reader = ConfluenceReader(base_url=base_url)
    documents = reader.load_data(
        space_key=space_key, include_attachments=True, page_status="current"
    )
    # documents = reader.load_data(
    #     cql=cql_query, include_attachments=True, max_num_results=20
    # )

    # print(cql)
    print(documents)
    for doc in documents:
        docs.append(doc.text)
        


    index = VectorStoreIndex.from_documents(
        documents, embed_model=get_embedding_function()
    )
    return index, docs



PROMPT_TEMPLATE = """ 
From given above context can you tell me how much percentage the response is matching with the query. Here is my Query and Response, \n
Query: {query}\nResponse: {response}\n
Provide your response like, Matching: [only percentage]
"""



def query_on_confluence_data_llamaindex(llm_model, index, documents, query_text):

    llm_model = verify_llm(llm_model)
    llm = Ollama(model=llm_model, request_timeout=1500.0)
    

    query_engine = index.as_query_engine(llm=llm)
    answer = query_engine.query(query_text)

    
    prompt = PROMPT_TEMPLATE.format(query=query_text, response=answer)

    match llm_model:
        case "mistral":
            LLM_MODEL_NAME = "llama2"
        case _:
            LLM_MODEL_NAME = "mistral"

    matchingIndex = text_to_query(LLM_MODEL_NAME, documents, prompt)
    matchingIndex = matchingIndex.body.decode("utf-8")

    response_text = {"response": answer.response, "hallucinatingPercentage": matchingIndex}

    return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)


