import os, time
from utilservice import *
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from services.embedding import get_embedding_function
from services.extractingResponse import extract_single_input


CHROMA_PATH = "./chroma/vectorDB"
ollamaModel = Ollama(model="mistral", keep_alive=-1)

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
    prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT)
    final_prompt = prompt_template.format(userPrompt=prompt, context=context_text)

    response_text = ollamaModel.invoke(final_prompt)
    language_code, language_name = language_detaction(response_text)
    response_obj = await extract_single_input(response_text)

    if not response_obj["theme"] or not response_obj["description"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"theme and description are not extracted from llm response")

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj

