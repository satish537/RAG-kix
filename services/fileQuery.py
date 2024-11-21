from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader, JSONLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms.ollama import Ollama
from fastapi.responses import JSONResponse
from fastapi import status
from utilservice import *
from time import sleep

DATA_PATH = "./data"
ollamaModel = Ollama(model="mistral", keep_alive=-1)

PROMPT_TEMPLATE = """
Here is some text:
{text}

Based on the above text, answer the following question:
{query}

"""

async def load_documents(data_path: str, filename: str):
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

async def text_to_query(filename: str, question: str):

    context = await load_documents(DATA_PATH, filename)
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(text=context, query=question)
    response_text = ollamaModel.invoke(prompt)

    return response_text
