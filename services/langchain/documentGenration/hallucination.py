import asyncio
import os, time
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate 
from services.langchain.embedding import get_embedding_function
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from utilservice import *


# Template for generating prompts to query the language model
PROMPT_TEMPLATE = """

CONTEXT: {context} 

--- 

I provide one prompt and response of that prompt as below.
Please analyze the response in relation to the provided context and determine the percentage of alignment between the response and the context.
Return the result in the format [X%]: 

PROMPT: {prompt} 

--- 

RESPONSE: {response} 

"""



# Path to the Chroma vector store
DATA_PATH = "./././data"
CHROMA_PATH = "./././chroma/langchain"


def getHallucination(llmModel: str, projectID: str, recordingID: str, promptObj: dict):

    path = f"{CHROMA_PATH}/{projectID}"
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=path, embedding_function=get_embedding_function())

    llmModel = verify_llm(llmModel)
    model = Ollama(model=llmModel)

    responseObj = responseGen(db, promptObj, model, recordingID)

    return responseObj


def responseGen(db, promptObj, model, recordingID):

    prompts_response = {}

    for category, prompt in promptObj.items():

        if recordingID:
            results = db.similarity_search_with_score(prompt['prompt'], k=7, filter={"recordingID": recordingID})
        else:
            results = db.similarity_search_with_score(prompt['prompt'], k=7)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = prompt_template.format(context=context_text, prompt=prompt['prompt'], response=prompt['response'])

        response_text = model.invoke(final_prompt)
        language_code, language_name = language_detaction(response_text)

        prompts_response[category] = {
            'data': response_text,
            'lanCode': language_code
        }

    return prompts_response






