import os, time
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from services.langchain.embedding import get_embedding_function
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from utilservice import *
from time import sleep
from services.langchain.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()

# Path to the Chroma vector store
CHROMA_PATH = "./././chroma/langchain"

# Template for generating prompts to query the language model
PROMPT_TEMPLATE = """
CONTEXT:    \n\n

{context}

---

Based on the provided context, which is segmented into chunks separated by "\n\n---\n\n," each chunk begins with metadata related to the query content. Responses should be concise and well-organized. Detailed responses should only be given upon request.

When a user includes metadata in their query, tailor the response to that specific metadata. 
Metadata includes a key called VersionID, representing the version of the document. If a user queries a specific version, provide the answer based on the corresponding VersionID appended at the beginning of the chunk. 
If metadata is not included in the question, base your response on the latest or most recently uploaded document.
Metadata present in the query but related content is not in context then just given answer "Not specific mention for this metadata."
If the data isn't found, don't create content that involves hallucinations. Instead, just provide the response "Not specific mention".

Respond in the same language as the provided context. Even if the user asks their question in a different language, your answer should match the language of the context.

Please answer the following question based on the context: {question}
"""



def chatAnswer(
    projectID: str, 
    llmModel: str,
    query_text: str,
    documentType: str
):
    context = extractSimilarityContent(projectID, query_text, documentType)
    response = answerRetriever(llmModel, context, query_text)

    return response


def extractSimilarityContent(project_id: str, query_text:str, documentType: str):

    path = f"{CHROMA_PATH}/{project_id}"
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{project_id}' Not Found")

    db = Chroma(persist_directory=path, embedding_function=get_embedding_function())
    
    # Fetch the similarity search results
    if documentType == "Input Documents":
        results = db.similarity_search_with_score(query_text, k=5, filter={"documentType": {"$in": ["User Document", "AI Generated Document"]}})
    elif documentType == "Output Documents":
        results = db.similarity_search_with_score(query_text, k=5, filter={"documentType": {"$in": ["User Document", "AI Generated Document"]}})
    else:
        results = db.similarity_search_with_score(query_text, k=5)

    # Combine the results into context text
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])

    return context_text


def answerRetriever(llmModel: str, context: str, query_text: str):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)

    response_text = ollama_model.invoke(prompt)

    language_code, language_name = language_detaction(response_text)
    response_obj = {
        "response": response_text,
        "language": language_name
    }

    return response_obj

