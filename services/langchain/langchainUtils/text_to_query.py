from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms.ollama import Ollama
from fastapi.responses import JSONResponse
from fastapi import status
from utilservice import *
from time import sleep
from services.langchain.langchainUtils.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()


PROMPT_TEMPLATE = """
Here is some text:
{text}

Based on the above text, answer the following question:
{query}

"""

def text_to_query(llm_model: str, text: str, query: str):

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(text=text, query=query)

    response_text = ollama_model.invoke(prompt)

    return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)
