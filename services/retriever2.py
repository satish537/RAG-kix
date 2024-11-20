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


SUPPORTING_TEXT_PROMPT = """

Analyze the provided transcript and extract the exact sentences or phrases that directly answer the question. Your task is to generate a list of supporting texts that strictly adheres to the following requirements:

Output Requirements:
Exact Match: The extracted content must be verbatim from the transcript and directly relevant to the question.
No Modifications: Do not add ellipses (...), parentheses, or any explanatory notes. The output must consist only of the sentences or phrases exactly as they appear in the transcript.
List Format: Provide the supporting texts in a list format, one item per line, without including the question or any additional text.
Clarity and Relevance: Each line must correspond to a unique, relevant point from the transcript, ensuring it answers the question.
No Redundancy: Avoid repeating the same sentence or phrase more than once.

Final Output Format:
[Exact matching text block from the transcript]
[Exact matching text block from the transcript]
[Exact matching text block from the transcript]

Input:

1. Question: {question}

===============================================================================================================================

2. Transcript: {context}

"""

TITLE_AND_DESCRIPTION_PROMPT = """

You are tasked with generating a structured response based on a given question and its supporting text. Please follow the steps below:

Input:
   - Question: {question}
   - Supporting Text: {supporting_text}

Output:
   - Answer: Generate a concise and accurate answer to the question using the information from the supporting text.
   - Theme Title: Create a clear and relevant theme title that encapsulates the main idea of both the supporting text and the answer.



Make sure to ensure clarity and coherence in the answer and theme title, maintaining alignment with the supporting text.

"""


async def generate_supporting_text(prompt: str, context_text: str):
    max_attempts = 5  # Maximum retries
    attempt = 0
    aggregated_points = []  # List to hold all points from all attempts

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}...for supporting text")
        
        prompt_template = ChatPromptTemplate.from_template(SUPPORTING_TEXT_PROMPT)
        final_prompt = prompt_template.format(question=prompt, context=context_text)
        response_text = ollamaModel.invoke(final_prompt)

        supporting_text_list = await extract_clean_points(response_text)
        supporting_text_list = await get_matching_strings(context_text, supporting_text_list)
        aggregated_points.extend(supporting_text_list)

        if len(supporting_text_list) >= 3:
            break

    if len(supporting_text_list) < 3:
        supporting_text_list = aggregated_points

    supporting_text_list = list(set(supporting_text_list))
    supporting_text_list = [element for element in supporting_text_list if element and element.strip()]

    return supporting_text_list 


async def generate_themeDescription_text(prompt: str, supporting_text_list: str):

    max_attempts = 3  # Maximum retries
    attempt = 0
    respontheme = None
    answerse_text = None

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}...for theme and description")

        prompt_template = ChatPromptTemplate.from_template(TITLE_AND_DESCRIPTION_PROMPT)
        final_prompt = prompt_template.format(question=prompt, supporting_text=supporting_text_list)
        response_text = ollamaModel.invoke(final_prompt)
        respontheme, answerse_text = await extract_theme_and_answer(response_text)

        if respontheme and answerse_text:
            break

    return respontheme, answerse_text





async def generate_theme_details(projectID: str, prompt: str, questionId, participantId):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    if questionId and participantId:
        results = db.similarity_search_with_score(prompt, k=3, filter={"$and": [{"questionId": questionId}, {"participantId": participantId}]})
    else:
        results = db.similarity_search_with_score(prompt, k=3, filter={"projectId": projectID})

    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"projectId '{projectID}' does not have any matching data")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    suppoting_text_list = await generate_supporting_text(prompt, context_text)
    respontheme, answerse_text = await generate_themeDescription_text(prompt, str(suppoting_text_list))

    response_obj = {
        "theme": respontheme,
        "description": answerse_text,
        "supporting texts": suppoting_text_list
    }

    language_code, language_name = language_detaction(str(response_obj))
    if not response_obj["theme"] or not response_obj["description"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"theme and description are not extracted from llm response")

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj



import re

async def extract_clean_points(text):
    # Regex to match points starting with *, ., -, or numbers (with or without a period or parenthesis after the number)
    regex = r"(?:(?:[-*.\s]*\d*[.)]?\s*)|(?:[*.\-]\s*))(.*)"
    
    # Extract matches
    matches = re.findall(regex, text)
    
    # Clean matches by removing surrounding quotes, whitespace, commas, and dots
    cleaned_points = [re.sub(r"[,.]", "", match.strip().strip('"\'"')) for match in matches]
    
    return cleaned_points


async def extract_theme_and_answer(response_text):
    theme_pattern = r"\*\*Theme Title:\*\*\s*(.+)"
    answer_pattern = r"\*\*Answer:\*\*\s*(.+)"

    theme_match = re.search(theme_pattern, response_text)
    theme_title = theme_match.group(1).strip() if theme_match else None

    answer_match = re.search(answer_pattern, response_text)
    answer = answer_match.group(1).strip() if answer_match else None

    return theme_title, answer


