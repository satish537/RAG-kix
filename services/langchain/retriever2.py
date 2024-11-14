import os, time, re
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
CONTEXT:    \n\n

{context}

---

Based on the provided context, which is segmented into chunks separated by "\n\n---\n\n," each chunk begins with metadata related to the query content. Responses should be concise and well-organized. Detailed responses should only be given upon request.

When a user includes metadata in their query, tailor the response to that specific metadata. 
Metadata includes a key called VersionID, representing the version of the document. If a user queries a specific version, provide the answer based on the corresponding VersionID appended at the beginning of the chunk. 
If metadata is not included in the question, base your response on the latest or most recently uploaded document.
If the data isn't found, don't create content that involves hallucinations. Instead, just provide the response "Not specific mention".

Respond in the same language as the provided context. Even if the user asks their question in a different language, your answer should match the language of the context.

Please answer the following question based on the context: {question}

"""


# Function to run a query against the Chroma vector store and return a response from the language model
def chat_query(agent: str, llm_model: str, query_text: str, project_id: str):
    start_time = time.time()
    path = f"{CHROMA_PATH}/{project_id}"
    if not os.path.exists(path):
        print(path, "Database not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail= f"Project_id '{project_id}' Not Found")
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=f"{CHROMA_PATH}/{project_id}", embedding_function=embedding_function)
    no_of_result = db._collection.count()
    print("no_of_result", no_of_result) 

    results = db.similarity_search_with_score(query_text, k=7)
    print(results)
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
    print(query_text)
    print(response_obj)

    return JSONResponse(content=response_obj, status_code=status.HTTP_200_OK)




def chat_query2(agent: str, llm_model: str, query_text: str, project_id: str):

    PROMPT_METADATA = """
    
    {query}

    From the above query please extract metadata like document name and version id 
    response give in list like ['document: media 1', 'document: media 2', 'version: 001']
    
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_METADATA)
    prompt = prompt_template.format(query=query_text)
    llm_model = verify_llm(llm_model)  
    model = Ollama(model=llm_model, keep_alive = -1)
    
    response_text = model.invoke(prompt)
    print(response_text)
    document_name, version_id = extract_documents_and_versions(response_text)

    return JSONResponse(content=[document_name, version_id], status_code=status.HTTP_200_OK)



def extract_documents_and_versions(text):
    # Patterns to match document names and version IDs
    document_pattern = r"'document:\s*([^']+?)'"
    version_pattern = r"'version:\s*([^']+?)'"
    
    # Extract all document names
    documents = re.findall(document_pattern, text)
    
    # Extract all version IDs
    versions = re.findall(version_pattern, text)
    
    # Return lists of documents and versions
    return documents, versions




