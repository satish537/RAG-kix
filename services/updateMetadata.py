import uuid, copy
from datetime import datetime
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from services.embedding import get_embedding_function

CHROMA_PATH = "./chroma/vectorDB"

async def store_chunks_with_metadata_dict(chunk_list: list[str], metadata_dict: dict):
    if not chunk_list:
        print("No chunks provided to store.")
        return False

    try:
        # Step 1: Convert each chunk into a Document with copied metadata
        document_chunks = []
        for chunk in chunk_list:
            doc_metadata = copy.deepcopy(metadata_dict)
            document_chunks.append(Document(page_content=chunk, metadata=doc_metadata))

        # Step 2: Assign UUIDs and timestamps
        document_chunks = await compute_ids(document_chunks)

        # Step 3: Store in Chroma
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        chunk_ids = [doc.metadata["id"] for doc in document_chunks]
        db.add_documents(document_chunks, ids=chunk_ids)
        db.persist()

        print(f"Stored {len(chunk_list)} chunks with metadata.")
        return True

    except Exception as e:
        print(f"Error storing chunks: {e}")
        return False

async def remove_documents_using_query(query_dict: dict):
    if not query_dict:
        print("No query conditions provided")
        return []

    db = Chroma(persist_directory=f"{CHROMA_PATH}", embedding_function=get_embedding_function())

    # Build the where clause - all conditions must match (AND logic)
    if len(query_dict) == 1:
        where_clause = query_dict
    else:
        conditions = [{key: value} for key, value in query_dict.items()]
        print(f"Conditions for deletion: {conditions}")
        where_clause = {"$and": conditions}

    try:
        matching_docs = db._collection.get(where=where_clause)
        print(f"Found {len(matching_docs.get('ids', []))} documents matching conditions: {query_dict}")
        # Extract matched document IDs and their page content
        doc_ids_to_delete = matching_docs.get('ids', [])
        page_contents = matching_docs.get('documents', [])  # documents hold the content chunks

        if doc_ids_to_delete:
            db._collection.delete(ids=doc_ids_to_delete)
            print(f"Deleted {len(doc_ids_to_delete)} documents matching conditions: {query_dict}")
        else:
            print(f"No documents found matching conditions: {query_dict}")

        db.persist()
        return page_contents  # Return the list of matched content chunks

    except Exception as e:
        print(f"Error deleting documents: {e}")
        return []

async def compute_ids(document_chunks: list[Document]) -> list[Document]:
    for doc in document_chunks:
        if "id" not in doc.metadata or not doc.metadata["id"]:
            doc.metadata["id"] = str(uuid.uuid4())
        if "timestamp" not in doc.metadata or not doc.metadata["timestamp"]:
            doc.metadata["timestamp"] = datetime.now().isoformat()
    return document_chunks


async def update_metadata_chunk(current_metadata: dict, new_metadata: dict) -> dict:
    
    chunk_list = await remove_documents_using_query(current_metadata)
    if not chunk_list:
        print("No documents found matching the current metadata.")
        return {}

    updated_count = len(chunk_list)

    await store_chunks_with_metadata_dict(chunk_list, new_metadata)
    
    return f"✅ Updated {updated_count} chunks with new metadata:\n{new_metadata}"
