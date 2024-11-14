# from services.langchain.text_to_query_test import text_to_query_test as langchain_query
from fastapi.responses import JSONResponse
from fastapi import status, HTTPException
from utilservice import *


def query_from_text_test(agent, llm_model, text, query, recordingId, categoryTitle, queryType):

    agent = verify_agent(agent)

    match agent:
        case "langchain":
            try:
                return langchain_query(llm_model, text, query, recordingId, categoryTitle, queryType)
            except Exception as e:
                return handel_exception(e)

        case _:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid option provided. Please choose from: langchain") 
