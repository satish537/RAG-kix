from utilservice import *
import os, docx2txt, uuid, copy
from datetime import datetime
from importlib.metadata import metadata
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from langchain.schema.document import Document
# from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from services.embedding import get_embedding_function
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader, JSONLoader

DATA_PATH = "./data"
CHROMA_PATH = "./chroma/vectorDB"


async def load_database2(uid, projectId, questionId, participantId, filename, videoType, metadata): 

    document_list = await load_documents(DATA_PATH, filename)
    chunk_list = await split_documents(document_list, uid, projectId, questionId, participantId, videoType, metadata)
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
async def split_documents(documents: list[Document], uid, projectId, questionId, participantId, videoType, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)

    for doc in split_docs:
        doc.metadata = metadata
        doc.metadata["uid"] = uid
        doc.metadata["projectId"] = projectId
        if questionId and participantId:
            doc.metadata["questionId"] = questionId
            doc.metadata["participantId"] = participantId
        doc.metadata["videoType"] = videoType 

    if split_docs:
        print("metadata", split_docs[0].metadata)
    else:
        print("No documents to split")

    return split_docs



async def add_to_db(chunk_list, projectId, DATA_PATH, filename):
    db = Chroma(
        persist_directory=f"{CHROMA_PATH}", embedding_function=get_embedding_function()
    )

    if chunk_list:
        newDocumentChunks = await compute_ids(chunk_list)
        new_chunk_ids = [chunk.metadata["id"] for chunk in newDocumentChunks]
        db.add_documents(newDocumentChunks, ids=new_chunk_ids)
        db.persist()
        delete_document(DATA_PATH, filename)
        return True
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This file has already been uploaded", )


async def compute_ids(chunks: list):
    unique_ids = set()  # Track generated IDs to ensure uniqueness

    for i in range(len(chunks)):
        unique_id = str(uuid.uuid4())

        # Ensure uniqueness even within the same batch
        while unique_id in unique_ids:
            unique_id = str(uuid.uuid4())

        unique_ids.add(unique_id)

        # Create a new metadata dictionary to prevent reference issues
        new_metadata = copy.deepcopy(chunks[i].metadata) if chunks[i].metadata else {}

        new_metadata["id"] = unique_id
        new_metadata["datetime"] = datetime.now().strftime("%Y%m%d%H%M%S%f")

        chunks[i].metadata = new_metadata  # Assign fresh metadata

    return chunks


