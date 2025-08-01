import json
from utilservice import *
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from services.load import load_database
from fastapi.responses import JSONResponse
from services.update import update_database
from services.summary import generate_summary
from services.OCR.main import extract_text_by_video
from services.retriever2 import generate_theme_details
from services.updateMetadata import update_metadata_chunk
from services.deleteVectors import removeDocumentsUsingQuery
from services.supportingText2 import generate_quotes_details
# from services.supportingText_test import generate_quotes_details_enhanced
from services.themeDescription import generate_theme_description_details
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form, status, BackgroundTasks, Body


app = FastAPI()
DATA_PATH = "data"




@app.post("/upload-transcript", tags=["Main"])
async def load_file(
    uid: str = Form(...), 
    projectId: str = Form(...), 
    questionId: str = Form(None), 
    participantId: str = Form(None), 
    metadata: str = Form(None),
    file: UploadFile = File(...), 
    videoType: str = Form(...),
):
    try:

        if metadata:
            try:
                metadata = json.loads(metadata)
                metadata = {k.lower(): v for k, v in metadata.items()}
            except json.JSONDecodeError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata format. Must be a valid JSON string.")
        else:
            metadata = {}

        fullpath, filename = await rename_and_save_file(file)
        response = await load_database(uid, projectId, questionId, participantId, filename, videoType, metadata)

        return JSONResponse(
            content="File Upload Successfully",
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
        delete_document(fullpath)



@app.post("/update-transcript", tags=["Main"])
async def update_file(
    uid: str = Form(...),
    projectId: str = Form(...),
    questionId: str = Form(None),
    participantId: str = Form(None),
    metadata: str = Form(None),
    file: UploadFile = File(...),
    videoType: str = Form(...),
):
    try:

        if metadata:
            try:
                metadata = json.loads(metadata)
                metadata = {k.lower(): v for k, v in metadata.items()}
            except json.JSONDecodeError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata format. Must be a valid JSON string.")
        else:
            metadata = {}

        fullpath, filename = await rename_and_save_file(file)
        response = await update_database(uid, projectId, questionId, participantId, filename, videoType, metadata)

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

    finally:
        delete_document(fullpath)




# Define request schema
class SourceInput(BaseModel):
    projectId: str
    uid: str
    metadata: Dict[str, Any] = Field(..., example={"videoType": "summary", "recordingId": "rec123"})

class ChunkStoreRequest(BaseModel):
    currentMetadata: SourceInput
    newMetadata: SourceInput

@app.post("/update-metadata", tags=["Main"])
async def store_chunks_from_sources(payload: ChunkStoreRequest):
    try:

        # Build metadata dicts
        metadata_1 = {
            "projectId": payload.currentMetadata.projectId,
            "uid": payload.currentMetadata.uid,
            **payload.currentMetadata.metadata
        }

        metadata_2 = {
            "projectId": payload.newMetadata.projectId,
            "uid": payload.newMetadata.uid,
            **payload.newMetadata.metadata
        }

        # Store both sets of chunks
        response = await update_metadata_chunk(metadata_1, metadata_2)

        return {"status": "success", "message": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing chunks: {str(e)}")




@app.post("/generate-theme", tags=["Main"])
async def generate_theme(projectId: str = Form(...), prompt: str = Form(...), questionId: str = Form(None), participantId: str = Form(None)):

    try:
    
        response = await generate_theme_details(projectId, prompt, questionId, participantId)

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



@app.post("/theme-description", tags=["Main"])
async def generate_theme_description(projectId: str = Form(...), prompt: str = Form(...), previousResponse: str = Form(None), questionId: str = Form(None), participantId: str = Form(None), metadata: str = Form(None), kValue: int = Form(3)):

    try:

        if metadata:
            try:
                metadata = json.loads(metadata)
                metadata = {k.lower(): v for k, v in metadata.items()}
            except json.JSONDecodeError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata format. Must be a valid JSON string.")
        else:
            metadata = {}
    
        response = await generate_theme_description_details(projectId, prompt, previousResponse, questionId, participantId, metadata, kValue)

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



@app.post("/quotes", tags=["Main"])
async def generate_supporting_text(
    projectId: str = Form(...),
    questionId: str = Form(None), 
    participantId: str = Form(None),
    prompt: str = Form(...),
    theme: str = Form(...),
    description: str = Form(...),
    metadata: str = Form(None),
    startKValue: int = Form(0),
    endKValue: int = Form(4),
    # chatId: str = Form(None),
    # callbackUrl: str = Form("https://aff1-2402-a00-167-2abe-3c39-2219-8605-5d2d.ngrok-free.app/api/canvas/ai-quotes-callback"),
):

    try:

        if metadata:
            try:
                metadata = json.loads(metadata)
                metadata = {k.lower(): v for k, v in metadata.items()}
            except json.JSONDecodeError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata format. Must be a valid JSON string.")
        else:
            metadata = {}

        response = await generate_quotes_details(projectId, questionId, participantId, prompt, theme, description, metadata, startKValue, endKValue)
        print(f"Generated quotes details: {response}")
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



@app.post("/generate-homework-summary", tags=["Main"])
async def generate_homework_summary(questionId: str = Form(...)):

    try:
    
        response = await generate_summary(questionId)

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




@app.post("/delete-vectors", tags=["Main"])
async def remove_documents_using_query(
    metadata: dict
):
    try:
        if not metadata:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No query conditions provided")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Metadata must be a dictionary")

        response = await removeDocumentsUsingQuery(metadata)

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


@app.post("/ocr-detector", tags=["Main"])
async def extract_text_using_ocr(
    file: UploadFile = File(...)
):
    try:
        fullpath, filename = await rename_and_save_file(file)
        results = await extract_text_by_video(fullpath)

        return JSONResponse(
            content=results,
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

    finally:
        file_path = os.path.join(DATA_PATH, file.filename)
        if os.path.isfile(file_path):
            delete_document(DATA_PATH, file.filename)



# @app.post("/quotes-test", tags=["Testing"])
# async def generate_supporting_text(
#     projectId: str = Form(...),
#     questionId: str = Form(None), 
#     participantId: str = Form(None),
#     prompt: str = Form(...),
#     theme: str = Form(...),
#     description: str = Form(...),
#     metadata: str = Form(None),
#     startKValue: int = Form(0),
#     endKValue: int = Form(4),
#     # chatId: str = Form(None),
#     # callbackUrl: str = Form("https://aff1-2402-a00-167-2abe-3c39-2219-8605-5d2d.ngrok-free.app/api/canvas/ai-quotes-callback"),
# ):

#     # try:

#     if metadata:
#         try:
#             metadata = json.loads(metadata)
#             metadata = {k.lower(): v for k, v in metadata.items()}
#         except json.JSONDecodeError:
#             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata format. Must be a valid JSON string.")
#     else:
#         metadata = {}

#     response = await generate_quotes_details_enhanced(projectId, questionId, participantId, prompt, theme, description, metadata, startKValue, endKValue)

#     return JSONResponse(
#         content=response,
#         status_code=status.HTTP_200_OK
#     )