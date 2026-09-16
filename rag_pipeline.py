import os
import re
import time
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

## These are created once when the module is imported, and reused
## across every request instead of being recreated each time.
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

VECTOR_STORE_ROOT = "faiss_indexes"

prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context:
    {context}

    Question:
    {question}
    """,
    input_variables=["context", "question"],
)


## Step A : Extract a video ID from any common YouTube URL format,
## or pass through a bare ID unchanged.
def extract_video_id(url_or_id: str) -> str:
    patterns = [
        r"(?:v=|\/shorts\/|youtu\.be\/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # If nothing matched, assume they already passed a bare video ID
    return url_or_id


## Step B : Getting Transcripts of the video (Document loader)
def get_transcript(video_id: str) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
        transcript = " ".join(snippet.text for snippet in fetched_transcript)
        return transcript
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")


## Step C : Helper to embed in small batches with delays, to avoid
## hitting Gemini's free-tier rate limit (429 RESOURCE_EXHAUSTED).
## Free tier = 100 embed requests/minute, and each chunk in a batch
## counts as its own request. batch_size=5 with delay=6s keeps us at
## (5 requests / 6s) * 60 = 50 requests/minute -- well under the cap,
## with retry-on-429 as a safety net in case we still get throttled.
def embed_in_batches(chunks, embeddings, batch_size=5, delay=6):
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        while True:
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    print("Rate limit hit, waiting 40s before retrying...")
                    time.sleep(40)
                else:
                    raise
        print(f"Embedded {i + len(batch)}/{len(chunks)} chunks")
        time.sleep(delay)
    return vector_store


## Step D : Splitting transcript into chunks + creating/loading the
## vector store. Loads from disk if this video was already indexed,
## so we never re-embed (and never re-burn quota) for the same video.
def get_or_create_vector_store(video_id: str, transcript: str) -> FAISS:
    vector_store_path = os.path.join(VECTOR_STORE_ROOT, video_id)

    if os.path.exists(vector_store_path):
        print(f"Loading existing vector store for {video_id}...")
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"Creating new vector store for {video_id}...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        vector_store = embed_in_batches(chunks, embeddings, batch_size=5, delay=6)
        os.makedirs(VECTOR_STORE_ROOT, exist_ok=True)
        vector_store.save_local(vector_store_path)

    return vector_store


## Step E : Build the retrieval + augmentation + generation chain for
## a given vector store, then run a question through it.
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(vector_store: FAISS):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser


def answer_question(vector_store: FAISS, question: str) -> str:
    chain = build_chain(vector_store)
    return chain.invoke(question)


## Step F : Convenience wrapper -- a summary is just a RAG answer to
## a fixed question.
def summarize(vector_store: FAISS) -> str:
    return answer_question(vector_store, "Can you summarize the key points from the video?")