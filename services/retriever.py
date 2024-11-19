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
ollamaModel = Ollama(model="mistral", keep_alive=-1)

REGENERATE_PROMPT = """

TITLE: {themeTitle}

Based on the provided theme title, extract supportive text from the transcript that aligns with the theme. 

Extract Supportive Text:
- Identify and quote specific sentences or phrases from the transcript that directly support the provided theme.
- Do not modify, paraphrase, or correct the text in any way. **Do not add commas or make grammatical corrections.**
- Include all grammatical errors, informal language, and typos as they appear in the original transcript.
- Verify Verbatim Text: Ensure the provided supportive text matches the transcript exactly, word-for-word. If the supportive text does not match verbatim, the response is invalid.
- Give minimum 3-5 points.

Supporting texts: - [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]

===========================================================================================================================================

TRANSCRIPT:
{context}


"""

CORE_PROMPT = """

QUESTION: {userPrompt}

You are tasked with analyzing a provided transcript and a related question. Your objectives are as follows:

Identify the Theme: Based on the question provided, extract and create a clear theme that encapsulates the essence of the question in relation to the transcript.

Provide a Description: Write a concise description (1-3 sentences) for the identified theme, explaining its relevance and context within the framework of the question and transcript.

Extract Supportive Text:

Identify and quote specific sentences or phrases from the transcript that directly support the theme.
Do not modify, paraphrase, or correct the text in any way. Don't add comma in text.
Include all grammatical errors, informal language, and typos as they appear in the original transcript.
Verify Verbatim Text: Ensure the provided supportive text matches the transcript exactly, word-for-word. If the supportive text does not match verbatim, the response is invalid.

Theme: [title of theme]\\Description: [description of theme]\\Supporting texts: - [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]
 
===========================================================================================================================================

TRANSCRIPT:
{context}

"""

async def regenerate_supporting_text(context_text: str, theme: str):
    prompt_template = ChatPromptTemplate.from_template(REGENERATE_PROMPT)
    final_prompt = prompt_template.format(themeTitle=theme, context=context_text)

    response_text = ollamaModel.invoke(final_prompt)

    return response_text

async def verify_supporting_text(response_obj: dict, context: str):
    supporting_texts_list = await get_matching_strings(context, response_obj['supporting texts'])
    matching_text = supporting_texts_list

    attempts = 0
    while len(supporting_texts_list) < 3 and attempts < 3:
        print("attempts", attempts)
        response = await regenerate_supporting_text(context, response_obj['theme'])
        supporting_texts_list = await extract_points_from_regenerated_res(response)

        supporting_texts_list = await get_matching_strings(context, supporting_texts_list)
        matching_text.extend(supporting_texts_list)
        attempts += 1

    if len(supporting_texts_list) < 3:
        supporting_texts_list = matching_text

    return supporting_texts_list


async def generate_theme_details(projectID: str, prompt: str, questionId, participantId):

    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    if questionId and participantId:
        results = db.similarity_search_with_score(prompt, k=2, filter={"$and": [{"questionId": questionId}, {"participantId": participantId}]})
    else:
        results = db.similarity_search_with_score(prompt, k=2, filter={"projectId": projectID})

    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"projectId '{projectID}' does not have any matching data")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT)
    final_prompt = prompt_template.format(userPrompt=prompt, context=context_text)

    response_text = ollamaModel.invoke(final_prompt)
    language_code, language_name = language_detaction(response_text)
    response_obj = await extract_single_input(response_text)

    supporting_text = await verify_supporting_text(response_obj, context_text)
    response_obj["supporting texts"] = supporting_text

    if not response_obj["theme"] or not response_obj["description"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"theme and description are not extracted from llm response")

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj

