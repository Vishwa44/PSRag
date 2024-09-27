import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.documents import Document
# from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import time
from collections import deque

app = FastAPI()

vector_store = None
ensemble_retriever = None
CHAT_HISTORY_LIMIT = 5
chat_history = deque(maxlen=CHAT_HISTORY_LIMIT)

def load_files():
    with open("all_rawtexts.pkl", "rb") as filehandler:
        all_rawtexts = pickle.load(filehandler)
    with open("document_list.pkl", "rb") as filehandler:
        document_list = pickle.load(filehandler)
    with open("uidlist.pkl", "rb") as filehandler:
        uidlist = pickle.load(filehandler)
    print("loading done")
    return all_rawtexts, document_list, uidlist

def query_openai_api(query: str, context: str, chat_history):
    history = ""
    print("chat_history len", len(chat_history))
    if chat_history:
        history = "\n\n".join([f"User: {item['query']}\nContext: {item['context']}" for item in chat_history])

    prompt = f"{history}\n\nUser question: {query}\n\nRelevant context: {context}\n\nAnswer based on the context:"
    # print("Prompt: ", prompt)
    client = OpenAI(api_key=os.environ.get('OPENAI_KEY'))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for the website PartSelect that answers questions based on provided context. Don't mention anything about the context when answering"},
            {"role": "user", "content": prompt}
        ]
    )
    return response

def initialize_retrievers():
    global vector_store, ensemble_retriever

    all_rawtexts, document_list, uidlist = load_files()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    vector_store.add_documents(documents=document_list, ids=uidlist)

    retriever_bm25 = BM25Retriever.from_documents(document_list)
    retriever_bm25.k = 2
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_bm25, faiss_retriever], weights=[0.5, 0.5])

    print("Retrievers initialized")

@app.on_event("startup")
def on_startup():
    initialize_retrievers()

class QueryInput(BaseModel):
    query: str

@app.post("/query")
async def query_api(query_input: QueryInput):
    global ensemble_retriever
    query_text = query_input.query

    if ensemble_retriever is None:
        return {"error": "Retrievers not initialized"}

    st = time.time()
    docs = ensemble_retriever.invoke(query_text)
    print("Retrieval time: ", time.time()-st)
    top_docs = [i.page_content for i in docs]

    context = "\n\n".join(top_docs)
    print(context)
    chat_history.append({
        "query": query_text,
        "context": context
    })

    st = time.time()
    openai_response = query_openai_api(query_text, context, chat_history)
    print("API call time: ", time.time()-st)
    # print(openai_response.choices[0].message.content)
    return {"result": openai_response.choices[0].message.content}
