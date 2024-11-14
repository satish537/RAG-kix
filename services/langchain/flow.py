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
from services.langchain.diagram import convert_to_uml
from services.langchain.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()

PROMPT_TEMPLATE = """
Identify speakers and their important topics (release, launch, decisions, tasks) in the following text, following this format: speaker: decision and make s>

{context}
"""


def flowtoquery(agent, llm_model, data_path, filename):

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
    documents = document_loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embedding_function = get_embedding_function()
    db = Chroma.from_documents(docs, embedding_function)
    results = db.similarity_search_with_score("", k=2)

    # Extract speaker and text from results
    speaker_data = []
    for i, (doc, _score) in enumerate(results):
        speaker = f"speaker{i+1}"
        text = doc.page_content
        speaker_data.append((speaker, text))

    context_text = "\n".join([f"{speaker}: {decision}\n{text}" for speaker, decision in speaker_data])
    uml=convert_to_uml(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)

    response_text = ollama_model.invoke(prompt)

    delete_document(data_path, filename)

    return JSONResponse(uml, status_code=status.HTTP_200_OK)
