import json
from typing import List
from utilservice import *
from enum import Enum
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from services.langchain.loadDocuments.transcriptDoc import transcriptToChunks
from services.langchain.loadDocuments.reqDoc import reqDoctoChunks
from services.langchain.loadDocuments.userDoc import userUploadedDocument
from services.langchain.dataCleanup.projectData import deleteProject, removeDocuments


router = APIRouter()
DATA_PATH = "./././data"


@router.delete("/project-delete/{projectID}")
async def remove_Documents_From_VectorDB(projectID: str):

    try:
        response = await deleteProject(projectID)

        return JSONResponse(
            content=f"The database for project {projectID} has been deleted.",
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



@router.delete("/record-delete/{projectID}/{recordingID}")
async def remove_Documents_From_VectorDB(projectID: str, recordingID: str):
    try:
        response = await removeDocuments(projectID, recordingID)

        return JSONResponse(
            content=f"All records with recordingID {recordingID} for project {projectID} have been deleted.",
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



    