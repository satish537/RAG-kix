import chromadb, os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from services.llamaindex.embedding import get_embedding_function
from llama_index.llms.ollama import Ollama
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *


DB_PATH = "././chroma/llamaindex"


def load_database(data_path, filename, project_id):
    documents = load_documents(data_path)
    return add_to_db(documents, data_path, project_id, filename)


def load_documents(data_path):
    document_loader = SimpleDirectoryReader(input_dir=f"././{data_path}").load_data()
    return document_loader


def add_to_db(documents, data_path, project_id, filename):

    db = chromadb.PersistentClient(path=f"{DB_PATH}/{project_id}")

    chroma_collection = db.get_or_create_collection(f"{project_id}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=get_embedding_function()
    )
    delete_document(data_path, filename)

    return JSONResponse(content="File Upload Successfully", status_code=status.HTTP_201_CREATED)





