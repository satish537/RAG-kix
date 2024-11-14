from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status


router = APIRouter()
DATA_PATH = "./././data"


