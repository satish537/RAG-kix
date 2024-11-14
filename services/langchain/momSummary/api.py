import json, time, requests
from enum import Enum
from utilservice import *
from typing import Optional, Dict
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from services.langchain.momSummary.momConcise import generate_mom_summary
from services.langchain.momSummary.momappLarge import generate_mom_summary_large
from services.langchain.momSummary.file_to_mom import generate_prompt
from services.langchain.momSummary.file_to_mom_test import generate_prompt as generate_prompt_test
from services.langchain.momSummary.text_to_mom import text_to_summary
from services.langchain.momSummary.momapp import generate_prompt_mom
from services.langchain.testing_stage.transcript_to_mom import generate_summary


router = APIRouter()
DATA_PATH = "./././data"


@router.post("/momapp/initiate", tags=["Main"])
async def initiate_mom_generation(
    background_tasks: BackgroundTasks,
    llm_model: str = Form(...),
    file: UploadFile = File(...),
    project_id: str = Form(...),
    media_duration: str = Form(...),
    recording_id: str = Form(...)
):
    start_time = time.time()
    fullpath, filename = await rename_and_save_file(file, "document", "0.0.1")
    background_tasks.add_task(handle_background_task, llm_model, DATA_PATH, filename, project_id, recording_id, start_time, media_duration)

    return {"status": "success", "message": "Request received"}


async def handle_background_task(llm_model, data_path, filename, project_id, recording_id, start_time, media_duration=50):
    media_duration = int(media_duration)
    if media_duration <= 60:
        result = await generate_mom_summary(llm_model, data_path, filename, recording_id)
        # result = await generate_prompt_mom(llm_model, data_path, filename, recording_id)
    else:
        result = await generate_mom_summary_large(llm_model, filename, project_id, recording_id)

    url = 'http://localhost:3000/api/recording/mom-callback'

    data = {
        "recordingId": recording_id,
        "response": json.dumps(result),
        "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
    }

    try:
        response = requests.post(url, data=data)
        execution_time = f"{time.time() - start_time:.2f}"
        print(f"MOM summary request sent to MoM API successfully for recording id : {recording_id} in {execution_time} seconds.", "\n")
    except Exception as e:
        print("----------------------------------------------------------------")
        print(e)


    return result




# @router.post("/momapp-large", tags=["Testing"])
# async def load_transcript(
#     projectID: str = Form(...), 
#     recordingId: str = Form(...), 
# ):
#     response = await generate_prompt("qwen2.5:14b", projectID, recordingId)

#     return JSONResponse(
#             content=response, 
#             status_code=status.HTTP_200_OK
#         )



# @router.post("/consis-mom", tags=["Testing"])
# async def consis_form_of_mom(llm_model: str = Form(...), file: UploadFile = File(...), recording_id: str = Form(...)):
#     print("file-to-mom API Called", recording_id)
#     fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
#     return await generate_mom_summary(llm_model, DATA_PATH, filename, recording_id)



# @router.post("/file-to-mom", tags=["Testing"])
# async def file_to_prompt(llm_model: str = Form(...), file: UploadFile = File(...), recording_id: str = Form(...)):
#     print("file-to-mom API Called", recording_id)
#     fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
#     return await generate_prompt(llm_model, DATA_PATH, filename, recording_id)



# @router.post("/file-to-mom-testing", tags=["Testing"])
# async def file_to_mom_testing(llm_model: str = Form(...), file: UploadFile = File(...), recording_id: str = Form(...)):

#     print(f"file-to-mom API Called for recording ID: {recording_id}")

#     # Extract filename and save the uploaded file
#     try:
#         fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

#     # Process the uploaded file and generate MoM summary
#     response = await generate_prompt_test(llm_model, DATA_PATH, filename, recording_id)

#     return response





# @router.post("/transcript-to-momsummary", tags=["Testing"])
# async def initiate_mom_generation(
#     file: UploadFile = File(...),
#     recording_id: str = Form(...),
#     media_duration: str = Form(...)
# ):
#     fullpath, filename = await rename_and_save_file(file, "document", "0.0.1")
#     media_duration = int(media_duration)
#     if media_duration > 0:
#         if media_duration <= 45:
#             print("text-to-mom called")
#             result = await text_to_summary(filename)
#         else:
#             print("momapp called")
#             result = await generate_prompt_mom("llama3", DATA_PATH, filename, recording_id)

#     return JSONResponse(content=result, status_code=status.HTTP_200_OK)


# @router.post("/transcript-to-summary", tags=["Testing"])
# async def initiate_mom_generation(
#     file: UploadFile = File(...),
# ):
#     fullpath, filename = await rename_and_save_file(file, "document", "0.0.1")
#     response = await generate_summary(filename, DATA_PATH)

#     return JSONResponse(content=response, status_code=status.HTTP_200_OK)









