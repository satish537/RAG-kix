import os
import shutil
from typing import List
from fastapi import status
from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from services.langchain.embedding import get_embedding_function



CHROMA_PATH = "./././chroma/langchain"

async def deleteProject(projectID: str):

    # Path to the database directory
    database_path = f"{CHROMA_PATH}/{projectID}"

    # Check if the database directory exists
    if not os.path.exists(database_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project database not found.")

    # Remove the database directory
    shutil.rmtree(database_path)

    return True




async def removeDocuments(projectID: str, recordingID: str):

    db = Chroma(persist_directory=f"{CHROMA_PATH}/{projectID}", embedding_function=get_embedding_function())
    
    matching_docs = db._collection.get(where={"recordingID": recordingID})
    doc_ids_to_delete = [doc['id'] for doc in matching_docs['metadatas']]
    
    print("Count before:", db._collection.count())
    if doc_ids_to_delete:
        db._collection.delete(ids=doc_ids_to_delete)
    print("Count after:", db._collection.count())
    
    db.persist()

    return True
           




