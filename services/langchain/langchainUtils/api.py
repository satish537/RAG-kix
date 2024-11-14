from utilservice import *
from pydantic import BaseModel
from typing import Optional, Dict
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from services.langchain.langchainUtils.text_to_query import text_to_query


router = APIRouter()
DATA_PATH = "./././data"


class TextToQuery(BaseModel):
    agent: str
    llm_model: str
    text_content: str
    query: str

@router.post("/text-to-query", tags=["Main"])
async def query(param: TextToQuery):
    response = text_to_query(param.llm_model, param.text_content, param.query)
    return response