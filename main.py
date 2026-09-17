from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import (
    extract_video_id,
    get_transcript,
    get_or_create_vector_store,
    answer_question,
    summarize,
)

app = FastAPI()

## Allow the React dev server (usually localhost:5173 or 3000) to call
## this API. Tighten this list once you know your frontend's real port.
## Allow the React dev server (usually localhost:5173 or 3000) to call
## this API. Tighten this list once you know your frontend's real port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+|https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    video_url: str


class ProcessResponse(BaseModel):
    video_id: str
    summary: str


class ChatRequest(BaseModel):
    video_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/api/process", response_model=ProcessResponse)
def process_video(request: ProcessRequest):
    video_id = extract_video_id(request.video_url)

    try:
        transcript = get_transcript(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = get_or_create_vector_store(video_id, transcript)
    summary = summarize(vector_store)

    return ProcessResponse(video_id=video_id, summary=summary)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    from rag_pipeline import VECTOR_STORE_ROOT, embeddings
    from langchain_community.vectorstores import FAISS
    import os

    vector_store_path = os.path.join(VECTOR_STORE_ROOT, request.video_id)
    if not os.path.exists(vector_store_path):
        raise HTTPException(
            status_code=404,
            detail="This video hasn't been processed yet. Call /api/process first.",
        )

    vector_store = FAISS.load_local(
        vector_store_path, embeddings, allow_dangerous_deserialization=True
    )
    answer = answer_question(vector_store, request.question)

    return ChatResponse(answer=answer)