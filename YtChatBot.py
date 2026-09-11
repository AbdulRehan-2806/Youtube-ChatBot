import os
import time
from dotenv import load_dotenv
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

## PART-1 : INDEXING
## Step 1 : Getting Transcripts of the video(Document loader)
## video_id = "Gfr50f6ZBvo"
video_id = "LPZh9BOjkQs"
try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
    transcript = " ".join(snippet.text for snippet in fetched_transcript)
    ## print(transcript)
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

## Step 2 : Splitting the transcript into chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))

## Step 3 : Creating embeddings for the chunks

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

## Step 3a : Helper to embed in small batches with delays, to avoid
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
                    ## print("Rate limit hit, waiting 40s before retrying...")
                    time.sleep(40)
                else:
                    raise
        ## print(f"Embedded {i + len(batch)}/{len(chunks)} chunks")
        time.sleep(delay)
    return vector_store

## Step 4 : Creating a vector store (load from disk if it already
## exists for this video, so we don't re-embed and burn quota every run)

VECTOR_STORE_PATH = "faiss_index_" + video_id

if os.path.exists(VECTOR_STORE_PATH):
    print("Loading existing vector store...")
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new vector store (this will call the embedding API)...")
    vector_store = embed_in_batches(chunks, embeddings, batch_size=10, delay=2)
    vector_store.save_local(VECTOR_STORE_PATH)

## print(vector_store.docstore._dict)

## PART-2 : RETRIEVAL

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

question = "What is the step 1 in training the large language model?"

docs = retriever.invoke(question)

# for doc in docs:
#     print(doc.page_content)
# for i, doc in enumerate(docs):
#     print(f"\n--- Document {i+1} ---")
#     print(doc.page_content)


## PART-3 : AUGMENTATION

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)

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

question = "What is the step 1 in training the large language model?"

docs = retriever.invoke(question)

context_text = "\n\n".join(
    [doc.page_content for doc in docs]
)

final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})

## PART-4 : GENERATION

answer = llm.invoke(final_prompt)

if isinstance(answer.content, list):
    for block in answer.content:
        if block.get("type") == "text":
            print(block["text"])
else:
    print(answer.content)