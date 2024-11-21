from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form, status, BackgroundTasks, Body
from services.load import load_database
# from services.retriever import generate_theme_details
from services.retriever2 import generate_theme_details
from services.retriever3 import generate_theme_details as generate_theme_details2
from services.summary import generate_summary
from services.fileQuery import text_to_query
from services.chatPrompt import retriveWithPrompt
from utilservice import *


app = FastAPI()
DATA_PATH = "data"




@app.post("/upload-transcript", tags=["Main"])
async def load_file(id: str = Form(...), projectId: str = Form(...), questionId: str = Form(None), participantId: str = Form(None), file: UploadFile = File(...), videoType: str = Form(...)):
    try:

        fullpath, filename = await rename_and_save_file(file)
        response = await load_database(id, projectId, questionId, participantId, filename, videoType)

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
        file_path = os.path.join(DATA_PATH, file.filename)
        if os.path.isfile(file_path):
            delete_document(DATA_PATH, file.filename)



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



@app.post("/generate-homework-summary", tags=["Main"])
async def generate_homework_summary(id: str = Form(...)):

    try:
    
        response = await generate_summary(id)

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



@app.post("/file-to-query", tags=["Main"])
async def load_file(file: UploadFile = File(...), question: str = Form(...)):
    try:

        fullpath, filename = await rename_and_save_file(file)
        response = await text_to_query(filename, question)

        return JSONResponse(
            content=response,
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
        file_path = os.path.join(DATA_PATH, file.filename)
        if os.path.isfile(file_path):
            delete_document(DATA_PATH, file.filename)



@app.post("/chat-with-prompt", tags=["Main"])
async def generate_theme(projectId: str = Form(...), prompt: str = Form(...), kValue: int = Form(...)):

    try:
    
        response = await retriveWithPrompt(projectId, prompt, kValue)

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



@app.post("/generate-theme-2", tags=["Main"])
async def generate_theme(projectId: str = Form(...), prompt: str = Form(...), questionId: str = Form(None), participantId: str = Form(None), kValue: int = Form(7)):

    try:
    
        response = await generate_theme_details2(projectId, prompt, questionId, participantId, kValue)

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

