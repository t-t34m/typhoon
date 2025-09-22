from fastapi import FastAPI, File, UploadFile
from langchain_openai import ChatOpenAI
import requests
import json
from typhoon_ocr import ocr_document
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()  # Load variables from .env into environment



app = FastAPI()

class TyphoonRequest(BaseModel):
    markdown: str
    prompt: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for testing, allow all; later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)
llm = ChatOpenAI(
    model="typhoon-v2.1-12b-instruct",
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=20000
)

@app.post("/OCR_image/")
async def analyze_video(file: Annotated[UploadFile, File()]):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".JPG") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    markdown = ocr_document(
        pdf_or_image_path=tmp_path,
        task_type="default"
    )
    return JSONResponse(content={"detail": markdown,})
@app.post("/ask_typhoon/")
async def ask_typhoon(req: TyphoonRequest):
    combined_prompt = f"{req.prompt.strip()}\n\n{req.markdown.strip()}"
    
    try:
        response = llm.invoke(combined_prompt)
        return JSONResponse(content={
            "prompt": req.prompt,
            "ocr_text": req.markdown,
            "typhoon_answer": response.content
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e)
        })
