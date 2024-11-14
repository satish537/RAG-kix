import asyncio
import os, time
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate 
from services.langchain.embedding import get_embedding_function
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from time import sleep
from utilservice import *
from services.langchain.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()

# Template for generating prompts to query the language model
PROMPT_TEMPLATE = """
Answer the question based only on the following context.

============================================================================================================================

Context:
{context}

============================================================================================================================


You are an advanced information retrieval assistant designed to extract the most recent and relevant information from a collection of interconnected documents. Each document is divided into chunks, and every chunk includes metadata indicating its date and time of upload. Your task is to provide a final answer to user queries based on the content available in these documents. 

When responding to a query, follow these guidelines:

1. Identify Relevant Documents: Analyze all documents in the dataset to identify which ones contain information related to the user's query.

2. Extract Information: For each relevant document, extract the pertinent information that addresses the user's question.

3. Determine Recency: Examine the metadata of each information chunk to determine the most recent content. Prioritize information from the chunk with the latest date and time.

4. Synthesize Final Answer: If the information is present in multiple documents, synthesize a coherent final answer based on the most recent chunk. Ensure that the response reflects the latest insights or conclusions drawn from the collective information.


Question:
{question} 

---

Do not add any additional phrases like 'Here is the response'. Strictly adhere to the following format:


{text_formate}

"""


PROMPT_TEMPLATE2 = """
Answer the question based only on the following context.

Context:
{context}

---

Question:
{question} 

---

Please provide response from the latest document based on the metadata.
Do not add any additional phrases like 'Here is the response'.

"""


# Path to the Chroma vector store
DATA_PATH = "./././data"
CHROMA_PATH = "./././chroma/langchain"


def document_prompts(llmModel: str, projectID: str, recordingID, promptObj: dict):

    path = f"{CHROMA_PATH}/{projectID}"
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=path, embedding_function=get_embedding_function())

    # llmModel = verify_llm(llmModel)
    # model = Ollama(model=llmModel)

    responseObj = responseGen(db, promptObj, recordingID)

    return responseObj


def responseGen(db, promptObj, recordingID):

    prompts_response = {}

    for category, prompt in promptObj.items():

        if recordingID:
            results = db.similarity_search_with_score(prompt['prompt'], k=5, filter={"recordingID": recordingID})
        else:
            results = db.similarity_search_with_score(prompt['prompt'], k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        if 'textContent' in prompt:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            final_prompt = prompt_template.format(context=context_text, question=prompt['prompt'], text_formate=prompt['textContent'])
        else:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE2)
            final_prompt = prompt_template.format(context=context_text, question=prompt['prompt'])

        response_text = ollama_model.invoke(final_prompt)
        language_code, language_name = language_detaction(response_text)

        prompts_response[category] = {
            'data': response_text,
            'lanCode': language_code
        }

    return prompts_response



