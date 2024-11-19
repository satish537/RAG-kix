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

QUESTION: {userQuestion}

Based on the provided question, extract supportive text from the transcript that aligns with the question.

Extract Supportive Text:
- Identify and quote specific sentences or phrases from the transcript that directly answer or support the provided question.
- Do not modify, paraphrase, or correct the text in any way. **Do not add commas or make grammatical corrections.**
- Include all grammatical errors, informal language, and typos as they appear in the original transcript.
- Verify Verbatim Text: Ensure the provided supportive text matches the transcript exactly, word-for-word. If the supportive text does not match verbatim, the response is invalid.
- Provide a minimum of 3-5 points.

Supporting texts: - [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]

===========================================================================================================================================

TRANSCRIPT:
{context}


"""

CORE_PROMPT = """
 
QUESTION: {userPrompt}
 
Identify a theme from the provided transcript and describe it concisely in 1-3 sentences.
Additionally, explain the rationale behind the title of the theme, ensuring to include exact phrases or sentences from the transcript that directly influenced your decision.When citing supporting text, please provide word-for-word excerpts from the transcript to substantiate your theme identification.
Generate supportive text that strictly matches the provided transcript.Ensure that every line corresponds 100% with the transcript without any grammatical mistakes.Do not include ellipses (...) or any additional text within parentheses or braces.The output must consist solely of lines that directly match the transcript, maintaining a perfect alignment with the original content.
 
Theme: [title of theme]\\Description: [description of theme]\\Supporting texts: - [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]- [exact matching text block from the transcript, ensuring it includes a point from the same context/transcript]
 
===========================================================================================================================================
 
TRANSCRIPT:
{context}
 
"""

async def regenerate_supporting_text(context_text: str, question: str):
    prompt_template = ChatPromptTemplate.from_template(REGENERATE_PROMPT)
    final_prompt = prompt_template.format(userQuestion=question, context=context_text)
    print(final_prompt)

    response_text = ollamaModel.invoke(final_prompt)

    return response_text

async def verify_supporting_text(response_obj: dict, context: str, question: str):
    supporting_texts_list = await get_matching_strings(context, response_obj['supporting texts'])
    matching_text = supporting_texts_list

    attempts = 0
    while len(supporting_texts_list) < 3 and attempts < 3:
        print("attempts", attempts)
        response = await regenerate_supporting_text(context, question)
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

    supporting_text = await verify_supporting_text(response_obj, context_text, prompt)
    response_obj["supporting texts"] = supporting_text

    if not response_obj["theme"] or not response_obj["description"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"theme and description are not extracted from llm response")

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj

