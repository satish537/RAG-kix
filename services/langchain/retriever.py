import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from services.langchain.embedding import get_embedding_function
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
# from chromadb.utils import parse_query_string
from utilservice import *


# Path to the Chroma vector store
CHROMA_PATH = "././chroma/langchain"


# Template for generating prompts to query the language model
PROMPT_TEMPLATE = """
Please respond to the following question based on the provided context. 
Your response must be in the same language as the context, regardless of the language of the question.

Context:
{context}

---

Question:
{question}

---

Please provide your answer in the language of the context.
"""



# Function to run a query against the Chroma vector store and return a response from the language model
def run_query(llm_model: str, query_text: str, project_id: str):
    path = f"{CHROMA_PATH}/{project_id}"
    if not os.path.exists(path):
        print("Database not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail= f"Project_id '{project_id}' Not Found")
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=f"{CHROMA_PATH}/{project_id}", embedding_function=embedding_function)
    no_of_result = db._collection.count()

    results = db.similarity_search_with_score(query_text, k=5)
    print(results)
    docANDver_results = []
    docORver_results = []

    # for result, _ in results:
    #     documentName = result.metadata.get("documentName")
    #     versionID = result.metadata.get("versionID")

    #     if find_word_in_sentence(documentName, query_text) and find_word_in_sentence(versionID, query_text):
    #         docANDver_results.append(result)
    #     elif find_word_in_sentence(documentName, query_text) or find_word_in_sentence(versionID, query_text):
    #         docORver_results.append(result)    

    # if docANDver_results != []:
    #     context_text = "\n\n---\n\n".join([doc.page_content for doc in docANDver_results])
    # elif docORver_results != []:
    #     context_text = "\n\n---\n\n".join([doc.page_content for doc in docORver_results])
    # else:
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    llm_model = verify_llm(llm_model)  
    model = Ollama(model=llm_model, keep_alive = -1)
    
    response_text = model.invoke(prompt)
    language_code, language_name = language_detaction(response_text)

    response_obj = {
        "response": response_text,
        "language": language_name
    }

    return JSONResponse(content=response_obj, status_code=status.HTTP_200_OK)


