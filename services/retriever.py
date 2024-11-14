import os, time
from utilservice import *
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from services.embedding import get_embedding_function


CHROMA_PATH = "./chroma/vectorDB"
ollamaModel = Ollama(model="mistral", keep_alive=-1)

CORE_PROMPT = """

QUESTION: {userPrompt}

Please identify a theme regarding above question from provided transcript along with a 1-3 sentence description which is explaining a theme and provide text from the above transcript without any modification to support a theme and text should be exact matching text block of transcript. please provide your response like, theme: title of theme then line feed description: description of theme then line feed supporting texts: list of supporting text which starts with - sign separated by line feed for theme.
 
===========================================================================================================================================

TRANSCRIPT:
{context}

"""


async def generate_theme_details(projectID: str, prompt: str):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    results = db.similarity_search_with_score(prompt, k=5, filter={"projectID": projectID})
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT)
    final_prompt = prompt_template.format(userPrompt=prompt, context=context_text)

    response_text = ollamaModel.invoke(final_prompt)
    language_code, language_name = language_detaction(response_text)

    response_obj = {
        'data': response_text,
        'lanCode': language_code
    }

    return response_obj