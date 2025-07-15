import os, time, re
from utilservice import *
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from services.embedding import get_embedding_function
from services.extractingResponse import get_matching_strings

CHROMA_PATH = "./chroma/vectorDB"
ollamaModel = Ollama(model="llama3.1", keep_alive=-1)



DESCRIPTION_PROMPT = """

You are tasked with generating a structured response based on a given question and its context. Please follow the steps below:

Input:
   - Question: {question}
   - Context Text: {context}

Output:
   - Answer: Generate a concise and accurate answer to the question using the information from the context text.

Make sure to ensure clarity and coherence in the answer and theme title, maintaining alignment with the context text.

"""


TITLE_PROMPT = """

You are an AI tasked with generating a concise and appropriate title based on the provided description.

Your response should follow this specific format: - Title: generated title

Ensure that the title is clear, impactful, and not overly lengthy.
The title should encapsulate the essence of the description while maintaining clarity and brevity.

**Description Input:** {context}

"""




async def generate_description_text(prompt: str, context_text: str, previousResponse):

    max_attempts = 3  # Maximum retries
    attempt = 0
    answerse_text = None

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}...for description")

        prompt_template = ChatPromptTemplate.from_template(DESCRIPTION_PROMPT)

        if previousResponse:
            context_text = f"{previousResponse}\n\n---\n\n{context_text}"

        final_prompt = prompt_template.format(question=prompt, context=context_text)
        response_text = ollamaModel.invoke(final_prompt)

        answerse_text = await extract_answer(response_text)

        if answerse_text:
            break

    return answerse_text


async def generate_theme_text(context_text: str):

    max_attempts = 5  # Maximum retries
    attempt = 0
    respontheme = None

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt}...for theme")

        prompt_template = ChatPromptTemplate.from_template(TITLE_PROMPT)
        final_prompt = prompt_template.format(context=context_text)

        response_text = ollamaModel.invoke(final_prompt)
        respontheme = await extract_theme(response_text)

        if respontheme:
            break

    return respontheme


async def build_chroma_filter(projectID=None, questionId=None, participantId=None, metadata=None):
    conditions = []

    # Add scalar fields
    if projectID:
        conditions.append({"projectId": projectID})
    if questionId:
        conditions.append({"questionId": questionId})
    if participantId:
        conditions.append({"participantId": participantId})

    # Add metadata fields (list or scalar)
    if metadata:
        for key, values in metadata.items():
            if isinstance(values, list):
                if len(values) == 1:
                    conditions.append({key: values[0]})
                elif len(values) > 1:
                    or_block = [{"%s" % key: v} for v in values]
                    conditions.append({"$or": or_block})
            else:
                conditions.append({key: values})

    # Return flat if only one condition, else wrap in $and
    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"$and": conditions}
    else:
        return {}  # no filters



async def generate_theme_description_details(projectID: str, prompt: str, previousResponse, questionId, participantId, metadata, kValue):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    filter_obj = await build_chroma_filter(
        projectID=projectID,
        questionId=questionId,
        participantId=participantId,
        metadata=metadata
    )

    results = db.similarity_search_with_score(prompt, k=kValue, filter=filter_obj)

    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"projectId '{projectID}' does not have any matching data")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    answerse_text = await generate_description_text(prompt, context_text, previousResponse)
    respontheme = await generate_theme_text(answerse_text)

    response_obj = {
        "theme": respontheme,
        "description": answerse_text
    }

    language_code, language_name = language_detaction(str(response_obj))
    if not response_obj["theme"] or not response_obj["description"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"theme and description are not extracted from llm response")

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj




async def extract_answer(response_text):
    answer_pattern = r"\*\*Answer:\*\*\s*([\s\S]+)"
    
    answer_match = re.search(answer_pattern, response_text)
    answer = answer_match.group(1).strip() if answer_match else None
    
    return answer


async def extract_theme(response_text):
    # Updated pattern to handle multiple formats
    theme_patterns = [
        r"(?:^|\n)[-*]?\s*(?:Title:|Title)\s*[\"']?(.*?)[\"']?(?:\n|$)",  # Handles bullet points and quotes
        r"\*\*Title:\*\*\s*[\"']?(.*?)[\"']?(?:\n|$)",                    # Handles markdown format
        r"^Title:\s*[\"']?(.*?)[\"']?(?:\n|$)"                            # Handles plain format
    ]

    for pattern in theme_patterns:
        theme_match = re.search(pattern, response_text, re.MULTILINE)
        if theme_match:
            theme_title = theme_match.group(1).strip()
            # Remove any remaining quotes if present
            if theme_title.startswith('"') and theme_title.endswith('"'):
                theme_title = theme_title[1:-1]
            elif theme_title.startswith("'") and theme_title.endswith("'"):
                theme_title = theme_title[1:-1]
            return theme_title

    return None




