import os, time
from utilservice import *
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from services.embedding import get_embedding_function
from services.extractingResponse import extract_single_input, get_matching_strings, extract_points_from_regenerated_res


CHROMA_PATH = "./chroma/vectorDB"
ollamaModel = Ollama(model="llama3.1", keep_alive=-1)


CORE_PROMPT = """
 
PROMPT:
{userPrompt}
 
=====================================================================================================================================================
 
TRANSCRIPT:
{context}
 
"""


async def retriveWithPrompt(projectID: str, prompt: str, kValue: int = 2):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    results = db.similarity_search_with_score(prompt, k=kValue, filter={"projectId": projectID})
    print(results)
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"projectId '{projectID}' does not have any matching data")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT)
    final_prompt = prompt_template.format(userPrompt=prompt, context=context_text)
    print("\n\n\n" ,"final_prompt")
    print(final_prompt)
    response_text = ollamaModel.invoke(final_prompt)
    language_code, language_name = language_detaction(response_text)

    response_obj = {
        'data': response_text,
        'lanCode': language_code
    }

    return response_obj

