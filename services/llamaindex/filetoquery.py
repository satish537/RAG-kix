import json 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *



Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def filetoquery(llm_model, data_path, filename, query):

    llm_model = verify_llm(llm_model)
    Settings.llm = Ollama(model=llm_model, request_timeout=1500.0)

    documents = SimpleDirectoryReader(input_files=[f"././data/{filename}"]).load_data()
    print("**documents**", documents)
    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine(streaming=True)
    response_text = query_engine.query(query)
    response_text = str(response_text)

    delete_document(data_path, filename)

    return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)