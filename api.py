import os, aiofiles, asyncio, json, requests, ast, time, threading
import docx                                                                                                             
import json
import whisperx
import torch
from typing import List
from pydantic import BaseModel
import requests
import wget
from utilservice import *
from deepmultilingualpunctuation import PunctuationModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form, status, BackgroundTasks, Body
from ollama_core.retriever import run_query
from ollama_core.flowtoquery import flow_to_query
from ollama_core.load_confluence import load_confluence
from ollama_core.text_to_query import query_from_text
from ollama_core.text_to_query_test import query_from_text_test
from services.langchain.flow import flowtoquery
from services.langchain.file_to_query_test import filetoquery as filetoquery_test
from services.llamaindex.confluence_to_vector import conf_to_vector
from services.langchain.doc_to_query import document_prompts
from services.langchain.image import imagesToPrompt
from services.langchain.highlights import process_mom_and_generate_videos
from services.langchain.audio_to_transcript1 import process_transcript_test
from services.langchain.extracted_doc import extract_headings_and_data,remove_data_under_headings,save_headings_to_json
from services.langchain.audio_trans_nemo_test import process_audio_file1
from services.langchain.audio_participants import process_audio_file
# from services.langchain.part1 import main
# from services.langchain.part1 import main
#from services.langchain.audio_trans_nemo_test1 import load_models
#from services.langchain.audio_trans_nemo_test1 import process_audio_file2
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf
from ctc_forced_aligner import load_alignment_model
from services.langchain.audio_to_transcript_test import process_transcript_test
from services.langchain.audio_whisper import process_audio_file_whisper
from services.langchain.audio_whisper import cleanup
from whisperx import load_model

# Routers
from services.langchain.loadDocuments.api import router as loadDocumentRouter
from services.langchain.documentGenration.api import router as documentGenrationRouter
from services.langchain.chatWithAI.api import router as chatWithAIRouter
from services.langchain.dataCleanup.api import router as dataCleanupRouter
from services.langchain.momSummary.api import router as momSummaryRouter
from services.langchain.confluence.api import router as confluenceRouter

app = FastAPI()
DATA_PATH = "././data"

API_KEY = "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="



# Include the router
app.include_router(loadDocumentRouter, prefix="")
app.include_router(documentGenrationRouter, prefix="")
app.include_router(chatWithAIRouter, prefix="")
app.include_router(momSummaryRouter, prefix="")
app.include_router(confluenceRouter, prefix="")
app.include_router(dataCleanupRouter, prefix="", tags=["Delete Operations"])



class query(BaseModel):
    agent: str
    llm_model: str
    query: str 
    project_id: str

@app.post("/ai", tags=["Main"])
async def get(param: query):
    return run_query(param.agent, param.llm_model, param.query, param.project_id)



@app.post("/document-and-prompts", tags=["Testing"])
async def document_to_prompts(llm_model: str = Form(...), file: UploadFile = File(...), recording_id: str = Form(...), prompt_jsonstr: str = Form(...)):
    
    try:
        prompt_obj = json.loads(prompt_jsonstr)

        if not isinstance(prompt_obj, dict):
            raise HTTPException(status_code=400, detail="Parsed JSON is not a dictionary")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON string")


    if os.path.splitext(file.filename)[0].endswith("_images"):
        print("images")
        fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
        response_obj = imagesToPrompt(llm_model, filename, recording_id, prompt_obj)
        return JSONResponse(content=response_obj, status_code=status.HTTP_200_OK)

    else:
        fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
        print(type(prompt_obj))
        return document_prompts(llm_model, DATA_PATH, filename, recording_id, prompt_obj)




@app.post("/file-to-query-test", tags=["Testing"])
async def file_to_prompt(agent: str = Form(...), llm_model: str = Form(...), file: UploadFile = File(...)):
    fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    return filetoquery_test(agent, llm_model, DATA_PATH, filename)










@app.post("/highlights", tags=["Testing"])
async def highlights(
    llm_model: str = Form(...),
    video_file: UploadFile = File(...),
    transcript_file: UploadFile = File(...),
    timestamps_file: UploadFile = File(...),
    recording_id: str = Form(...),
):
    try:
        # Save the uploaded files to the server
        video_path, video_filename = await rename_and_save_file(video_file, document_name="video", version_id=recording_id)
        transcript_path, transcript_filename = await rename_and_save_file(transcript_file, document_name="transcript", version_id=recording_id)
        timestamp_path, timestamps_filename = await rename_and_save_file(timestamps_file, document_name="timestamps", version_id=recording_id)

        # Process the MoM and generate videos and images
        response = await process_mom_and_generate_videos(llm_model, DATA_PATH, video_filename, transcript_filename, timestamps_filename, recording_id)

        return response

    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": str(e)})

# @app.on_event("startup")
# async def startup_event():
#     global whisper_model
#     whisper_model = whisper.load_model("base")
#     print("Whisper model loaded.")

torch.cuda.empty_cache()

#def get_device_for_model():
#    gpu_count = torch.cuda.device_count()
    # Use only GPUs 0, 1, 2, and 3 if there are at least 4 GPUs available
#    if gpu_count >= 4:
#        return {"device": "cuda", "device_index": [0, 1, 2, 3]}
#    elif gpu_count > 1:
#        return {"device": "cuda", "device_index": list(range(gpu_count))}
#    elif gpu_count == 1:
#        return {"device": "cuda", "device_index": 0}
#    else:
#        return {"device": "cpu", "device_index": 0}  # Fallback to CPU if no GPUs

#device_config = get_device_for_model()
#whisperx_model = whisperx.load_model("large", language="en", device=device_config["device"], device_index=device_config["device_index"], compute_type="float16")

#@app.post("/audio-to-transcript-test", tags=["Testing"])
#async def initiate_audio_transcription(
#    background_tasks: BackgroundTasks,
#    file: UploadFile = File(...),
#   recording_id: str = Form(...)
#):
#    start_time = time.time()
#    print("audio-to-transcript-test endpoint called for recording_id:", recording_id)
    # Save the uploaded file
#    audiopath, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")

    # Add transcription task to the background (NON-BLOCKING)
#    background_tasks.add_task(handle_transcription_background_task, DATA_PATH, audiopath, recording_id, start_time)

    # Return a success response immediately while the transcription is processed in the background
#    return {"status": "success", "message": "Transcription request received"}

# Background task to handle transcription process
#async def handle_transcription_background_task(data_path, audiopath, recording_id, start_time):
#    global whisperx_model

#    try:
        # Process transcription in the background (LONG-RUNNING TASK)
        # print("Processing transcription")
#        result = await process_transcript_test(whisperx_model, data_path, audiopath, recording_id)

        # Callback URL to notify that transcription is complete
#        url = 'http://localhost:3000/api/recording/transcript-callback'

#        data = {
#            "recordingId": recording_id,
#            "response": json.dumps(result),
#            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
#        }

#        print("audio-to-transcript-test endpoint request complate successfully for recording _id:", recording_id,  "time taken:", time.time() - start_time)

        # Send the result to the callback URL
#        response = requests.post(url, data=data)
        # print(f"Transcription result sent to API successfully for recording id: {recording_id}")

#    except Exception as e:
#        print(f"Error processing transcription for recording id {recording_id}: {e}")
# @app.post("/audio-to-transcript-test", tags=["Testing"])
# async def audio_to_transcript_endpoint_1(file: UploadFile = File(...), recording_id: str = Form(...)):
#     # Pass the loaded model to the function
#     global whisperx_model
#     whisperx_model = whisperx.load_model("large", device="cuda", compute_type="float32")
#     try:
#         audiopath, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

#     response = await process_transcript_test(whisperx_model, DATA_PATH, audiopath, recording_id)
#     return response

class QueryConfluence(BaseModel):
    agent: str
    llm_model: str
    query: str
    url: str
    username: str
    api_key: str
    space_key: str

@app.post("/query-confluence", tags=["Main"])
async def confluence_load(param: QueryConfluence):
    response = load_confluence(param.agent, param.llm_model, param.query, param.url, param.username, param.api_key, param.space_key)
    return response


class ConfluenceToVector(BaseModel):
    url: str
    username: str
    api_key: str
    space_key: str

@app.post("/confluence-to-vector", tags=["Main"])
async def confluence_to_vector(param: ConfluenceToVector):
    return conf_to_vector(param.url, param.username, param.api_key, param.space_key)






class TextToQuery(BaseModel):
    agent: str
    llm_model: str
    text_content: str
    query: str

@app.post("/text-to-query", tags=["Main"])
async def query(param: TextToQuery):
    response = query_from_text(param.agent, param.llm_model, param.text_content, param.query)
    return response



class TextToQueryTest(BaseModel):
    agent: str
    llm_model: str
    text_content: str
    query: str
    recordingId: str
    categoryTitle: str
    queryType: str

@app.post("/text-to-query-test")
async def query(param: TextToQueryTest):
    print("Received Request from front end", param.recordingId)
    print("Sending response: {'status': 'success', 'message': 'Request received'}, status_code: 200")
    response = query_from_text_test(param.agent, param.llm_model, param.text_content, param.query,
                                            param.recordingId, param.categoryTitle, param.queryType)
    async def process_query():
        try:
    #        print(f"Query processing completed for recording ID {param.recordingId}: {response}")
            # Include callback response with additional fields
            callback_response = {
                "recordingId": param.recordingId,
                "categoryTitle": param.categoryTitle,
                "queryType": param.queryType,
                "response": response,
                "apiKey": API_KEY
                }
            print("Callback response:", callback_response)
        except Exception as e:
            print(f"Error during query processing for recording ID {param.recordingId}: {e}")


    loop = asyncio.get_event_loop()
    loop.create_task(process_query())

    # Immediately return a 200 OK response
    return response



@app.post("/flow-to-query-test", tags=["Testing"])
async def file_to_prompt(agent: str = Form(...), llm_model: str = Form(...), file: UploadFile = File(...)):
    fullpath, filename = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    result = flowtoquery(agent, llm_model, DATA_PATH, filename)
    return result


@app.post("/document_extract", tags=["Testing"])                                                                         
async def document(file: UploadFile = File(...)):
    documentName = "doc"
    versionID = "0.0.1"
    fullpath, filename = await rename_and_save_file(file, documentName, versionID)                                          
    headings = extract_headings_and_data(fullpath)
    headfile = remove_data_under_headings(fullpath, headings)
    save_headings_to_json(headings, 'extracted_headings_final.json')
    return headfile





@app.post("/process_word_document")
async def process_word_document(file: UploadFile = File(...), json_path: str = Form("headings.json")):
    docx_path=json_path.replace(".json",".docx").strip()
    json_path=json_path.strip()
    print("path")
    print(docx_path)
    print(json_path)
    with open(docx_path, "wb") as f:
        f.write(await file.read())
    headings = extract_headings_and_data(docx_path)
    remove_data_under_headings(docx_path, headings)
    save_headings_to_json(headings, json_path)
    return {
        "headings": headings,
        "message": f"Data removed successfully and headings saved to {json_path}",
        "updated_document": docx_path
    }

#@app.post("/audio-trans-nemo1")
#async def audio_to_transcript_endpoint_1(file: UploadFile = File(...), recording_id: str = Form(...)):
     # Pass the loaded model to the function
#    global whisperx_model
#     whisperx_model = whisperx.load_model("large", device="cuda", compute_type="float32")
#     try:
#        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
#     except Exception as e:
#        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

#     output_dir="/home/azureuser/"
#     response = process_audio_file(audio_path, output_dir, enable_stemming=True, batch_size=8, suppress_numerals=True)
#     return response



def get_device_for_model():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        return {"device": "cuda", "device_index": list(range(gpu_count))}
    elif gpu_count == 1:
        return {"device": "cuda", "device_index": 0}
    else:
        return {"device": "cpu", "device_index": 0}

device_config = get_device_for_model()
whisperx_model = whisperx.load_model("large", language="en", device=device_config["device"], device_index=device_config["device_index"], compute_type="float16")

@app.post("/transcribe_with_whisper", tags=["Testing"])
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    recording_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    try:
        # Save the uploaded audio file using rename_and_save_file
        audio_path, audiofile = await rename_and_save_file(audio_file, document_name="document", version_id="0.0.1")

        print("Audio file saved, starting transcription task.")

        # Add transcription task to the background (NON-BLOCKING)
        background_tasks.add_task(handle_transcription_background_task_nemo, audio_path, recording_id)

        # Return a success response immediately while the transcription is processed in the background
        return JSONResponse(content={"status": "success", "message": "Transcription request received"})

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")
        
# Background task to handle transcription process
async def handle_transcription_background_task_nemo(audio_path, recording_id):
    try:
        result = await process_transcript_test(whisperx_model, DATA_PATH, audio_path, recording_id)

        # Handle callback or further processing here if needed
        callback_url = 'http://localhost:3000/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(result),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="  # Replace with your actual API key
        }

        # Send the result to the callback URL
        requests.post(callback_url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")

@app.post("/transcribe_with_whisper_test")
async def audio_to_transcript_endpoint_whisper(
    file: UploadFile = File(...),
    recording_id: str = Form(...)
):
    try:
        # Save the uploaded file
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    try:
        result = await process_transcript_test(whisperx_model, DATA_PATH, audio_path, recording_id)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing audio: {str(e)}")

    return result



@app.post("/audio-trans-nemo1")
async def audio_to_transcript_endpoint_1(
    file: UploadFile = File(...),
    participants: str = Form(...),
    recording_id: str = Form(...)
):
    try:
        # Save the uploaded file
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    # Split the participants string into a list
    participant_list = [name.strip() for name in participants.split(',')]

    output_dir = "/home/azureuser/"
    try:
        # Process the audio file synchronously, passing participants to the function
        response = process_audio_file(
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True,
            participants=participant_list  # Include participants in processing
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing audio: {str(e)}")

    return response

@app.post("/audio-trans-video")
async def audio_to_transcript_endpoint_1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: str = Form(...)
):
    try:
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    # Add transcription task to the background (NON-BLOCKING)
    background_tasks.add_task(handle_transcription_background_task1, audio_path, recording_id)

    # Return a success response immediately while the transcription is processed in the background
    return {"status": "success", "message": "Transcription request received"}

# Background task to handle transcription process
async def handle_transcription_background_task1(audio_path, recording_id):
    try:
        output_dir = "/home/azureuser/"
        response = process_audio_file1(  # Remove whisper_model parameter
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True
        )

        # Callback URL to notify that transcription is complete
        url = 'https://app.hapie.ai/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")
    


@app.post("/audio-transcribe")
async def audio_transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    participants: str = Form(None),
    recording_id: str = Form(...)
):
    try:
        # Save the uploaded file
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")

        # Split the participants string into a list
        participant_list = [name.strip() for name in participants.split(',')]

        # Add transcription task to the background (NON-BLOCKING)
        background_tasks.add_task(
            handle_transcription_background_task,
            audio_path,
            recording_id,
            participant_list
        )

        # Return a success response immediately while the transcription is processed in the background
        return {"status": "success", "message": "Transcription request received"}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error processing request: {str(e)}")

async def handle_transcription_background_task(audio_path: str, recording_id: str, participants: List[str]):
    try:
        output_dir = "/home/azureuser/"
        response = process_audio_file(
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True,
            participants=participants
        )

        # Callback URL to notify that transcription is complete
        url = 'http://localhost:3000/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")

# @app.post("/participants", tags=["Main"])
# async def audio_to_transcript_endpoint_1(
#     file: UploadFile = File(...),
#     participants: str = Form(...),
#     recording_id: str = Form(...)
# ):
#     try:
#         # Save the uploaded file
#         audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

#     # Split the participants string into a list
#     participant_list = [name.strip() for name in participants.split(',')]
#     print(participant_list)
#     #try:
#     # Process the audio file synchronously, passing participants to the function
#     response = main(
#         audio_path,
#         participants=participant_list  # Include participants in processing
#     )
#     #except Exception as e:
#     #    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing audio: {str(e)}")

#     return response

from services.langchain.audio_trans_nemo_test1 import AudioTranscriber
from moviepy.editor import VideoFileClip

models_loaded = False
output_dir = "output"
transcriber = None

@app.on_event("startup")
async def load_models_on_startup():
    global models_loaded, transcriber
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    load_models()  # Load models once when the server starts
    models_loaded = True

def load_models():
    global transcriber
    transcriber = AudioTranscriber(output_dir)

@app.post("/audio-trans-video-test")
async def audio_to_transcript_endpoint_1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: str = Form(...),
):
    try:
        # Rename and save the uploaded file
        audio_path = await rename_and_save_file1(file, document_name="document", version_id="0.0.1")

        # Check the file extension and handle accordingly
        if audio_path.endswith('.mp4'):
            audio_temp_path = audio_path.replace('.mp4', '.wav')  # Define a temporary WAV file path
            extract_audio_from_video(audio_path, audio_temp_path)  # Extract audio to WAV
            audio_path = audio_temp_path  # Update audio_path to point to the WAV file
        elif not audio_path.endswith('.mp3'):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file format. Please upload MP3 or MP4 files.")

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    if not models_loaded:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Models not loaded yet.")

    background_tasks.add_task(handle_transcription_background_task2, audio_path, recording_id)

    return {"status": "success", "message": "Transcription request received"}

async def handle_transcription_background_task2(audio_path, recording_id):
    try:
        # Call process_audio_file with the correct arguments
        response = transcriber.process_audio_file(audio_path, output_dir)

        # Call to the external callback URL
        url = 'http://localhost:3000/api/recording/transcript-callback'  # Update this URL as needed
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="  # Replace with your actual API key
        }

        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")

def extract_audio_from_video(video_path: str, audio_output_path: str):
    """Extract audio from video file."""
    try:
        with VideoFileClip(video_path) as video:
            audio = video.audio
            audio.write_audiofile(audio_output_path, codec='pcm_s16le')  # Use pcm_s16le codec
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error extracting audio: {str(e)}")

async def rename_and_save_file1(file: UploadFile, document_name: str, version_id: str):
    """Rename and save the uploaded file."""
    # Define the path where the file will be saved
    file_path = f"data/{document_name}_{version_id}_{file.filename}"

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return file_path  # Return only the file path as a string


@app.post("/audio-trans-video-whisper")
async def audio_to_transcript_endpoint_1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: str = Form(...)
):
    try:
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    # Add transcription task to the background (NON-BLOCKING)
    background_tasks.add_task(handle_transcription_background_task_whisper, audio_path, recording_id)

    # Return a success response immediately while the transcription is processed in the background
    return {"status": "success", "message": "Transcription request received"}

# Background task to handle transcription process
async def handle_transcription_background_task_whisper(audio_path, recording_id):
    try:
        timestamp = int(time.time())
        output_dir = f"/home/azureuser/{recording_id}_{timestamp}/"
        os.makedirs(output_dir, exist_ok=True)

        response = process_audio_file_whisper(
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True,
        )

        # Callback URL to notify that transcription is complete
        url = 'https://app.hapie.ai/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")

    finally:
        # Cleanup the entire output directory after processing is complete
        if output_dir:
            cleanup(output_dir)
