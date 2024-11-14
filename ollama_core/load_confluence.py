from services.langchain.load_confluence import get_confluence_data_as_vector_langchain, query_on_confluence_data_langchain
from services.llamaindex.load_confluence import get_confluence_data_as_vector_llamaindex, query_on_confluence_data_llamaindex
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *



def load_confluence(agent, llm_model, query, url, username, api_key, space_key):    

    agent = verify_agent(agent)

    match agent:
        case "langchain":
            try:
                index = get_confluence_data_as_vector_langchain(url, query, username, api_key, space_key)
                response = query_on_confluence_data_langchain(llm_model, index, query)
                return response
            except Exception as e:
                return handel_exception(e)

        case "llamaindex":
            try:
                index, documents = get_confluence_data_as_vector_llamaindex(url, llm_model, query, username, api_key, space_key)
                response = query_on_confluence_data_llamaindex(llm_model, index, documents, query)
                return response
            except Exception as e:
                return handel_exception(e)

        case _:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid option provided. Please choose from: langchain, llamaindex")
        
