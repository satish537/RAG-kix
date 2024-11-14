import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader, JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from services.langchain.embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *


# Path to the Chroma database
DB_PATH = "././chroma/langchain"


# Main function to load and process documents, and add them to the database
def load_database(data_path, filename, project_id, metadata, documentName, versionID):
    documents = load_documents(data_path, filename)      # Load documents from file
    chunks = split_documents(documents, metadata)         # Split documents into chunks
    return add_to_db(chunks, project_id, data_path, filename, documentName, versionID)          # Add chunks to the database


# Function to load documents based on their file extension
def load_documents(data_path, filename):
    _, file_extension = os.path.splitext(filename)
    match file_extension:
        case '.txt':
            document_loader = TextLoader(f"{data_path}/{filename}")
        case '.docx' | '.doc':
            document_loader = Docx2txtLoader(f"{data_path}/{filename}")
        case '.pdf':
            document_loader = PyPDFLoader(f"{data_path}/{filename}")
        case '.csv':
            document_loader = CSVLoader(f"{data_path}/{filename}")
        case '.xlsx':
            # document_loader = UnstructuredExcelLoader(f"{data_path}/{filename}", mode="single")
            excel_to_json(data_path, filename)
            document_loader = JSONLoader(
                file_path=f"{data_path}/{filename.split('.')[0]}.json",
                jq_schema='.[]',
                text_content=False
                )
        case _:
            print("File Format is Not Supported")
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File Format is Not Supported")
    return document_loader.load()


# Function to split documents into smaller chunks
def split_documents(documents: list[Document], metadata: dict):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)

    for doc in split_docs:
        meta_info = f"Metadata: {metadata}\n\n"
        doc.page_content = meta_info + doc.page_content
    print(split_docs)
    print(len(split_docs))
    return split_docs


# Function to add document chunks to the database
def add_to_db(document_chunks: list[Document], project_id, data_path, filename, documentName, versionID):
    db = Chroma(
        persist_directory=f"{DB_PATH}/{project_id}", embedding_function=get_embedding_function()
    )
    chunks_with_ids = compute_ids(document_chunks, project_id, filename, documentName, versionID)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        delete_document(data_path, filename)
        return JSONResponse(content="File Upload Successfully", status_code=status.HTTP_201_CREATED)
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This file has already been uploaded", )
    

# Function to compute unique IDs for document chunks
def compute_ids(chunks, project_id, filename, documentName, versionID):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}:{project_id}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
        chunk.metadata["documentName"] = documentName
        chunk.metadata["versionID"] = versionID
    return chunks
