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
ollamaModel = Ollama(model="llama3.1", keep_alive=-1)

CORE_PROMPT = """
 
You are an AI summarization tool. Your task is to extract the major points from the provided transcript. Please follow these guidelines:

Read the entire transcript carefully.
Identify and focus on the key ideas, arguments, and conclusions presented.
Summarize the content in a point-wise format.
Ensure each point is objective, concise, and focuses on the most significant information.
Each point should represent a general overview of the content and context, avoiding conversational tones or references to specific individuals (e.g., "you," "they," "he," "she").
Separate each point with a newline character("\n") and index numbers.
Avoid any unnecessary details, examples, or explanations; summarize only the major points and overarching themes.
 
===========================================================================================================================================

CONTEXT:
{context}

"""


async def generate_summary(questionId: str):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    results = db.similarity_search_with_score(CORE_PROMPT, k=10, filter={"questionId": questionId})
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"questionId '{questionId}' does not have any matching data")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT)
    final_prompt = prompt_template.format(userPrompt=CORE_PROMPT, context=context_text)

    response_obj = []
    attempt = 0
    while not response_obj and attempt < 3:
        print(f"Attempt {attempt}...for summary")
        response_text = ollamaModel.invoke(final_prompt)
        print(response_text)
        response_obj = await extract_clean_key_points(response_text)

        attempt += 1


    language_code, language_name = language_detaction(str(response_obj))

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj



async def extract_clean_key_points(text):
    # Define a regex pattern to match points starting with keymarks (numbers, dashes, asterisks)
    pattern = r"^(?:\d+\.\s*|-{1,2}\s*|\*\s*)(.*)"
    # Use re.findall with the multiline flag to extract the text without keymarks
    points = re.findall(pattern, text, flags=re.MULTILINE)
    # Strip whitespace from each extracted point
    points = [point.strip() for point in points]
    return points

