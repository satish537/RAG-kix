from services.langchain.text_to_query import text_to_query as langchain_query
from fastapi.responses import JSONResponse
from fastapi import status, HTTPException
from utilservice import *



def query_from_text(agent, llm_model, text, query):
    
    agent = verify_agent(agent)

    match agent:
        case "langchain":
            try:
                return langchain_query(llm_model, text, query)
            except Exception as e:
                return handel_exception(e)

        case _:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid option provided. Please choose from: langchain")
