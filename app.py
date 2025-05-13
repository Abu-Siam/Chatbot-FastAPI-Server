# app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import embed_csv, get_retriever
import os
import shutil

app = FastAPI()

# CORS setup (optional for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LLM and prompt
llm = OllamaLLM(model="gemma2:2b")
prompt = ChatPromptTemplate.from_template("""
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
""")
chain = prompt | llm

retriever = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global retriever
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embed_csv(file_path)
    retriever = get_retriever()

    return {"status": "success", "message": "CSV embedded successfully."}


@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    global retriever
    if retriever is None:
        return {"error": "Please upload a CSV first."}

    relevant_docs = retriever.invoke(payload.question)
    reviews = "\n".join([doc.page_content for doc in relevant_docs])
    answer = chain.invoke({"reviews": reviews, "question": payload.question})

    return {"answer": answer}
