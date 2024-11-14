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
Analyze the following context, which includes both text and table data. 
If context data contain data in square brace so it is data of tables. 
Each table is organized with a [column header, row header, value] format, preceded by the table name.
 
For any queries regarding table data, refer to this structured format. If the answer isn't explicitly mentioned in the text or table data, respond with "Not Mentioned." Do not infer or predict answers from the table data.

Context:
{context}

---

Question:
{question}

---

Response must be in the English language as the context, if user want to in any specific language then change. 
Ensure a comprehensive response based on the provided information.
Response must be in Human language form, In the context provided all type od braces and backslash is for programming language and llm model. Don't given in Response.
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


