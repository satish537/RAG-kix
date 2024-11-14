import chromadb, os, shutil
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.readers.confluence import ConfluenceReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from fastapi.responses import JSONResponse
from fastapi import status, HTTPException
from utilservice import *


DATABASE_PATH = f"././chroma/confluence"

load_dotenv()


def conf_to_vector(url, username, api_key, space_key):
    try: 
        print("confluence to vector api call")

        os.environ["CONFLUENCE_USERNAME"] = username
        os.environ["CONFLUENCE_PASSWORD"] = api_key
        base_url = url
        collection_name = sanitize_collection_name(space_key)
        COLLECTION_PATH = f"{DATABASE_PATH}/{collection_name}"

        content = []


        # Remove Old Data
        if os.path.isdir(DATABASE_PATH):
            if os.path.isdir(f"{DATABASE_PATH}/{collection_name}"):
                shutil.rmtree(f"{DATABASE_PATH}/{collection_name}")

        # Create Collection Name Folder
        if not os.path.exists(f"{DATABASE_PATH}/{collection_name}"):
            os.makedirs(f"{DATABASE_PATH}/{collection_name}")
            
        # Read Confluence Data
        reader = ConfluenceReader(base_url=base_url)

        # Convert into Document object
        documents = reader.load_data(
            space_key=space_key, include_attachments=True, page_status="current"
        )


        for doc in documents:
            content.append(doc.text)

        # Define the file path
        file_path = os.path.join(COLLECTION_PATH, f"{collection_name}.txt")

        # Write content to the file
        with open(file_path, "w") as file:
            file.writelines(content)



        # Create Chromadb object
        db = chromadb.PersistentClient(path=f"{COLLECTION_PATH}/{collection_name}")

        chroma_collection = db.get_or_create_collection(f"{collection_name}")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Emabed model for Vector
        embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model
        )

        return JSONResponse(content="Data Stored Successfully", status_code=status.HTTP_201_CREATED)
    
    except Exception as error:
        return handel_exception(error)
