import chromadb, os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from services.llamaindex.embedding import get_embedding_function
from llama_index.llms.ollama import Ollama
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from utilservice import *



CHROMA_PATH = "././chroma/llamaindex"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def run_query(llm_model: str, query_text: str, project_id: str):

    llm_model = verify_llm(llm_model)
    Settings.llm = Ollama(model=llm_model, request_timeout=1500.0)

    path = f"{CHROMA_PATH}/{project_id}"
    if not os.path.exists(path):
        print("Database not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail= f"Project_id '{project_id}' Not Found")

    embedding_function = get_embedding_function()

    db = chromadb.PersistentClient(path=f"{CHROMA_PATH}/{project_id}")

    chroma_collection = db.get_collection(f"{project_id}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embedding_function
    )

    query_engine = index.as_query_engine()
    response_text = query_engine.query(query_text)

    return JSONResponse(content=response_text.response, status_code=status.HTTP_200_OK)




