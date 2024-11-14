import json
from enum import Enum
from utilservice import *
from pydantic import BaseModel
from typing import Optional, Dict
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from services.langchain.chatWithAI.chatDoc import chatAnswer
from services.langchain.retriever2 import chat_query
from services.langchain.retriever3 import run_query


router = APIRouter()
DATA_PATH = "./././data"


class documentTypeCatagory(str, Enum):
    inputdocuments = "Input Documents"
    outputdocuments = "Output Documents"


@router.post("/chat-ai", tags=["Main"])
async def load_transcript(
    projectID: str = Form(...), 
    llmModel: str = Form(...), 
    query: str = Form(...), 
    documentType: documentTypeCatagory = Form(...)
):
    try:

        response = chatAnswer(projectID, llmModel, query, documentType)

        return JSONResponse(
            content=response, 
            status_code=status.HTTP_200_OK
        )

    except HTTPException as http_exc:
        return JSONResponse(
            content=http_exc.detail,
            status_code=http_exc.status_code
        )

    except Exception as e:
        print(e)

        return JSONResponse(
            content=f"Internal server error: {str(e)}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class query(BaseModel):
    agent: str
    llm_model: str
    query: str 
    project_id: str


@router.post("/chat-with-ai", tags=["Testing"])
async def ChatWithAI(param: query):
    return chat_query(param.agent, param.llm_model, param.query, param.project_id)

class query2(BaseModel):
    llm_model: str
    query: str 
    project_id: str

@router.post("/chat-with-ai2", tags=["Testing"])
async def ChatWithAI2(param: query2):
    return run_query(param.llm_model, param.query, param.project_id)





