from services.langchain.flow import flowtoquery as langchain_query
# from services.llamaindex.flow import flowtoquery as llamaindex_query
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *


def flow_to_query(agent, llm_model, data_path, filename):

    agent = verify_agent(agent)

    match agent:
        case "langchain":
            try:
                return langchain_query(llm_model, data_path, filename)
            except Exception as e:
                delete_document(data_path, filename)
                return handel_exception(e)

        # case "llamaindex":
        #     try:
        #         return llamaindex_query(llm_model, data_path, filename)
        #     except Exception as e:
        #         delete_document(data_path, filename)
        #         return handel_exception(e)

        case _:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid option provided. Please choose from: langchain, llamaindex") 

