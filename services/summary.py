import os, time
from utilservice import *
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from services.embedding import get_embedding_function
from services.extractingResponse import extract_points_from_text


CHROMA_PATH = "./chroma/vectorDB"
ollamaModel = Ollama(model="mistral", keep_alive=-1)

CORE_PROMPT = """
 
You are an AI summarization tool. Your task is to extract the major points from the provided transcript. Please follow these guidelines:  

1. Read the entire transcript carefully.  
2. Identify and focus on the key ideas, arguments, and conclusions presented.  
3. Summarize the content in a point-wise format.  
4. Each point should be concise and only include the most significant information.  
5. Separate each point with a newline character "\n".  
6. Avoid any unnecessary details, examples, or explanations; stick to the major points only.  
 
===========================================================================================================================================

TRANSCRIPT:
{context}

"""


async def generate_summary(id: str):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    results = db.similarity_search_with_score(CORE_PROMPT, k=10, filter={"id": id})
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"ID '{id}' does not have any matching data")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT)
    final_prompt = prompt_template.format(userPrompt=CORE_PROMPT, context=context_text)

    response_text = ollamaModel.invoke(final_prompt)
    language_code, language_name = language_detaction(response_text)
    response_obj = await extract_points_from_text(response_text)

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj

