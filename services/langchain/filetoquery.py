import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from services.langchain.embedding import get_embedding_function
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *
from services.langchain.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()


# Template for generating prompts to query the language model
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# Function to load a document, split it into chunks, and run a query against the Chroma vector store
def filetoquery(llm_model, data_path, filename, query):

    # Determine the file extension and select the appropriate document loader
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
        case _:
            print("File Format is Not Supported")
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File Format is Not Supported")

    # Load the document
    documents = document_loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print("**docs**",docs)

    # Initialize Chroma vector store with the document chunks
    embedding_function = get_embedding_function()
    db = Chroma.from_documents(docs, embedding_function)
    no_of_result = db._collection.count()

    # Perform a similarity search with the query
    results = db.similarity_search_with_score(query, k=no_of_result)
    print("**results**", results)

    # Generate context text from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    response_text = ollama_model.invoke(prompt)

    # Delete the uploaded document
    delete_document(data_path, filename)
    
    # Return the response as a JSON response
    return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)