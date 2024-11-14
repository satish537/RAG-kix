from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form, status, BackgroundTasks, Body
from services.load import load_database
from services.retriever import generate_theme_details
from utilservice import *


app = FastAPI()
DATA_PATH = "data"




@app.post("/upload-transcript", tags=["Main"])
async def load_file(id: str = Form(...), projectId: str = Form(...), file: UploadFile = File(...), videoType: str = Form(...)):
    try:

        fullpath, filename = await rename_and_save_file(file)
        response = await load_database(id, projectId, filename, videoType)

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
        delete_document(DATA_PATH, file.filename)



@app.post("/generate-theme", tags=["Main"])
async def generate_theme(projectId: str = Form(...), prompt: str = Form(...)):

    try:
    
        response = await generate_theme_details(projectId, prompt)
        print(response)
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
