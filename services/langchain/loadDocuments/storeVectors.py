import os, uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from services.langchain.embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *


# Path to the Chroma database
DB_PATH = "./././chroma/langchain"
DATA_PATH = "./././data"


# Main function to load and process documents, and add them to the database
def load_database(
    document_objects: list, 
    projectID: str, 
    metadata: dict
):
    response = add_to_db(document_objects, projectID, metadata)

    return response


# Function to add document chunks to the database
def add_to_db(
    documentChunks: list[Document], 
    projectID: str, 
    metadata: dict
):
    db = Chroma(
        persist_directory=f"{DB_PATH}/{projectID}", embedding_function=get_embedding_function()
    )

    newDocumentChunks = compute_ids(documentChunks, projectID, metadata)

    new_chunk_ids = [chunk.metadata["id"] for chunk in newDocumentChunks]
    db.add_documents(newDocumentChunks, ids=new_chunk_ids)
    db.persist()

    return True

    

# Function to compute unique IDs for document chunks
def compute_ids(chunks: list, projectID: str, metadata: dict):
    
    for chunk in chunks:
        unique_id = str(uuid.uuid4())
        chunk.metadata["id"] = unique_id

        for key, value in metadata.items():
            chunk.metadata[f"{key}"] = value

    return chunks



