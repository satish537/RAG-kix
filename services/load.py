import os, docx2txt
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader, JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
from langchain_core.documents import Document
from services.embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *

DATA_PATH = "./data"
CHROMA_PATH = "./chroma/vectorDB"


async def load_database(id, projectId, filename, videoType):

    document_list = await load_documents(DATA_PATH, filename)
    chunk_list = await split_documents(document_list, id, projectId, videoType)
    response = await add_to_db(chunk_list, projectId, DATA_PATH, filename)

    return response


# Function to load documents based on their file extension
async def load_documents(data_path, filename):
    _, file_extension = os.path.splitext(filename)
    match file_extension:
        case '.txt':
            document_loader = TextLoader(f"{data_path}/{filename}")
        case '.docx' | '.doc':
            document_loader = Docx2txtLoader(f"{data_path}/{filename}")
        case '.pdf':
            document_loader = PyPDFLoader(f"{data_path}/{filename}")
        case _:
            print("File Format is Not Supported")
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File Format is Not Supported")
    return document_loader.load()


# Function to split documents into smaller chunks
async def split_documents(documents: list[Document], id, projectId, videoType):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)

    for doc in split_docs:
        doc.metadata["id"] = id
        doc.metadata["projectId"] = str(projectId)
        doc.metadata["videoType"] = videoType

    return split_docs



async def add_to_db(chunk_list, projectId, DATA_PATH, filename):
    db = Chroma(
        persist_directory=f"{CHROMA_PATH}", embedding_function=get_embedding_function()
    )

    if len(chunk_list):
        db.add_documents(chunk_list)
        db.persist()
        delete_document(DATA_PATH, filename)
        return True
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This file has already been uploaded", )



