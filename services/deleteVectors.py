from langchain_community.vectorstores import Chroma
from services.embedding import get_embedding_function

CHROMA_PATH = "./chroma/vectorDB"

async def removeDocuments(projectID: str):

    db = Chroma(persist_directory=f"{CHROMA_PATH}", embedding_function=get_embedding_function())
    
    matching_docs = db._collection.get(where={"projectId": projectID})
    doc_ids_to_delete = [doc['id'] for doc in matching_docs['metadatas']]
    
    if doc_ids_to_delete:
        db._collection.delete(ids=doc_ids_to_delete)
        print(f"Deleted {len(doc_ids_to_delete)} documents from project {projectID}")

    db.persist()

    return True


async def removeDocumentsUsingQuery(query_dict: dict):

    if not query_dict:
        print("No query conditions provided")
        return True
    
    db = Chroma(persist_directory=f"{CHROMA_PATH}", embedding_function=get_embedding_function())
    
    # Build the where clause - all conditions must match (AND logic)
    if len(query_dict) == 1:
        # Single condition
        where_clause = query_dict
    else:
        # Multiple conditions - use $and operator
        conditions = [{key: value} for key, value in query_dict.items()]
        where_clause = {"$and": conditions}
    
    try:
        matching_docs = db._collection.get(where=where_clause)
        doc_ids_to_delete = [doc['id'] for doc in matching_docs['metadatas']]
        
        if doc_ids_to_delete:
            db._collection.delete(ids=doc_ids_to_delete)
            print(f"Deleted {len(doc_ids_to_delete)} documents matching conditions: {query_dict}")
        else:
            print(f"No documents found matching conditions: {query_dict}")
        
        db.persist()
        return True
        
    except Exception as e:
        print(f"Error deleting documents: {e}")
        return False

