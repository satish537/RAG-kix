import json
from enum import Enum
from utilservice import *
from pydantic import BaseModel
from typing import Optional, Dict
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from services.langchain.documentGenration.documentGenrator import document_prompts
from services.langchain.documentGenration.query_retriever import run_query
from services.langchain.documentGenration.prompt_gen import prompt_template


router = APIRouter()
DATA_PATH = "./././data"


class PromptGenerator(BaseModel):
    llm_model: str
    input_string: str
    prompt: str

@router.post("/prompt-gen", tags=["Main"])
async def promptgen(param: PromptGenerator):
    return prompt_template(param.llm_model, param.input_string, param.prompt)


class documentTypeCatagory(str, Enum):
    inputdocuments = "Input Documents"
    outputdocuments = "Output Documents"

@router.post("/document-generator", tags=["Main"])
async def genrateAnswerofPrompt(
    llmModel: str = Form(...), 
    projectID: str = Form(...), 
    recordingID: Optional[str] = Form(None),
    promptObj: str = Form(...), 
):
    try:

        try:
            prompt_obj = json.loads(promptObj)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format"
            )
        
        response = document_prompts(llmModel, projectID, recordingID, prompt_obj)

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



@router.post("/document-hallucination/", tags=["Testing"])
async def genrateAnswerofPrompt(
    llmModel: str = Form(...), 
    projectID: str = Form(...), 
    recordingID: Optional[str] = Form(None),
    promptObj: str = Form(...), 
):
    try:
        prompt_obj = json.loads(prompt_obj)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON format"
        )



# class multiple_query_payload(BaseModel):
#     llm_model: str
#     prompt_obj: dict
#     project_id: str
#     recording_id: str

# @app.post("/multiple_query", tags=["Main"])
# async def process_queries(param: multiple_query_payload):

#     return run_query(param.llm_model, param.prompt_obj, param.project_id, param.recording_id)





