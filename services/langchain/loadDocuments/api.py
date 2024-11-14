import json
from typing import List
from utilservice import *
from enum import Enum
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from services.langchain.loadDocuments.transcriptDoc import transcriptToChunks
from services.langchain.loadDocuments.reqDoc import reqDoctoChunks
from services.langchain.loadDocuments.userDoc import userUploadedDocument
from services.langchain.loadDocuments.file_to_vector import load_database
from ollama_core.upload_file import upload_file
# from services.langchain.load2 import load_database
from services.langchain.testing_stage.load import load_database

router = APIRouter()
DATA_PATH = "./././data"


#  Load Transcript in vector database
class documentTypeCatagory(str, Enum):
    transcript = "Transcript"
    userDocument = "User Document"
    aiGenDocument = "AI Generated Document"

@router.post("/load-document", tags=["Main"])
async def load_Document_into_VectorDB(
    file: UploadFile = File(...), 
    projectID: str = Form(...), 
    recordingID: str = Form(None),
    templateID: str = Form(None),
    versionID: str = Form(None),
    templateUpdate: str = Form("False"),
    documentType: documentTypeCatagory = Form(...)
):
    
    try:
        fullpath, filename = await rename_and_save_file(file)
        filepath = f"{DATA_PATH}/{filename}"


        if documentType == "Transcript":
            metadata = {
                "recordingID": recordingID,
                "documentType": "Transcript"
            }
            response = await transcriptToChunks(filepath, projectID, metadata)

        elif documentType == "User Document":
            metadata = {
                "recordingID": recordingID,
                "documentType": "User Document"
            }
            response = userUploadedDocument(filepath, projectID, metadata)

        elif documentType == "AI Generated Document":
            metadata = {
                "templateID": templateID,
                "versionID": versionID,
                "documentType": "AI Generated Document"
            }
            response = await reqDoctoChunks(filepath, projectID, metadata, templateUpdate)


        if not response: 
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to load transcript")

        return JSONResponse(
            content="Transcript File Load Successfully",
            status_code=status.HTTP_201_CREATED
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

    finally:
        delete_document(DATA_PATH, filename)



# @router.post("/convert_to_vector")
# async def create_vector_representation(project_id: str = Form(...), recording_id: str = Form(...), file: UploadFile = File(...)):
#     fullpath, filename = await rename_and_save_file(file)
#     print(filename)
#     return await load_database(filename, project_id, recording_id)




@router.post("/upload", tags=["Testing"], summary="Endpoint to upload a file and process it for vector storage.", description="This endpoint allows users to upload a file, which is then processed and stored in the ChromaDB database. The file is converted into a vector representation suitable for use in a RAG application, enabling enhanced document retrieval and generation capabilities.")
async def load_file(agent: str = Form(...), project_id: str = Form(...), file: UploadFile = File(...), metadata: str = Form(...)):
    metadata = json.loads(metadata)
    if isinstance(metadata, dict):
        documentName = metadata.get("documentName", "doc")  # Set a default value
        versionID = metadata.get("versionID", "0.0.1")  # Set a default value
    else:
        documentName = "doc"
        versionID = "0.0.1"
    fullpath, filename = await rename_and_save_file(file, documentName, versionID)
    
    return upload_file(agent, DATA_PATH, filename, project_id, metadata, documentName, versionID)





@router.post("/upload_document", tags=["Testing"])
async def load_document(project_id: str = Form(...), file: UploadFile = File(...)):
    # metadata = json.loads(metadata)
    # if isinstance(metadata, dict):
    #     documentName = metadata.get("documentName", "doc")  # Set a default value
    #     versionID = metadata.get("versionID", "0.0.1")  # Set a default value
    # else:
    documentName = "doc"
    versionID = "0.0.1"
    fullpath, filename = await rename_and_save_file(file, documentName, versionID)
    
    return load_database(DATA_PATH, filename, project_id, documentName, versionID)






