from services.langchain.load import load_database as langchain_ld
from services.llamaindex.load import load_database as llama_index_ld
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from utilservice import *



def upload_file(agent, data_path, filename, project_id, metadata, documentName, versionID):
        
    agent = verify_agent(agent)

    match agent:
        case "langchain":
            try:
                response_text = langchain_ld(data_path, filename, project_id, metadata, documentName, versionID)
                return response_text
            except Exception as e:
                delete_document(data_path, filename)
                return handel_exception(e)

        case "llamaindex":
            try:
                response_text = llama_index_ld(data_path, filename, project_id)
                return response_text
            except Exception as e:
                delete_document(data_path, filename)
                return handel_exception(e)

        case _:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid option provided. Please choose from: langchain, llamaindex")
                