from services.langchain.retriever import run_query as langchain_query
from services.llamaindex.retriever import run_query as llamaindex_query
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from utilservice import *



def run_query(agent, llm_model, query_text, project_id):    

    agent = verify_agent(agent)

    match agent:
        case "langchain":
            try:
                return langchain_query(llm_model, query_text, project_id)
            except Exception as e:
                return handel_exception(e)

        case "llamaindex":
            try:
                return llamaindex_query(llm_model, query_text, project_id)
            except Exception as e:
                return handel_exception(e)

        case _:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid option provided. Please choose from: langchain, llamaindex")
