import os
import sys
import streamlit as st
from utils import logger
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms.llamacpp import LlamaCpp

# Import app configuration
from config import api_key, result_threshold, score_threshold, openai_embedding_model, openai_inference_models, local_models, llm_temperature, n_ctx, max_tokens, n_gpu_layers

sys.setrecursionlimit(4096)

replaceable=st.empty()

def save(documents: dict, vectorstore: dict):
    global replaceable

    fs=LocalFileStore("./cache/")

    # load cached embedding
    cached_embedder=CacheBackedEmbeddings.from_bytes_store(
        vectorstore['embedding'], fs, namespace=vectorstore['embedding'].model
    )

    vdb=FAISS.from_documents(documents, embedding=cached_embedder)
    try:
        vdb.save_local(vectorstore['path'])
    except Exception as e:
        logger(f"An error occured trying to reindex vector store: {e}", "error")
        return False

    return True

def load(vectorstore: dict):
    global replaceable

    vs_path=vectorstore['path']
    vs_embedding=vectorstore['embedding']

    vdb=FAISS.load_local(vs_path, embeddings=vs_embedding)

    return vdb

def get_vectorstore(url: str, model: str):
    global replaceable
    vectorstore={}

    logger(f"Creating vectorstore for {url} using {model}", "info")

    # openai vectorstore
    if model in openai_inference_models:

        # check if OPENAI_API_KEY is set
        if api_key == "":
            logger("Error: environment variable OPENAI_API_KEY is not set", "error")
            return None

        vectorstore['type']="openai"
        vectorstore['embedding']=OpenAIEmbeddings(
            openai_api_key=api_key,
            model=openai_embedding_model
        )
        vectorstore['name']=f"{url}-{openai_embedding_model}.vdb"

    # local vectorstore
    elif model in local_models:

        model_path=os.path.join(os.path.dirname(__file__), "models", model + ".gguf")
        # check if model_path exists
        if not os.path.exists(model_path):
            logger(f"Error: {model} model does not exist", "error")
            return None

        vectorstore['type']="local"
        vectorstore['embedding']=LlamaCppEmbeddings(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )
        vectorstore['name']=f"{url}-{model}.vdb"
    else:
        logger(f"Error: {model} model does not exist", "error")
        return None

    vectorstore['model']=model
    vectorstore['path']=os.path.join(os.path.dirname(__file__), "data", vectorstore['name'])
    return vectorstore

def search(question: str, vectorstore: dict):
    global replaceable

    logger(f"Loading vectorstore {vectorstore}", "info")

    try:
        vdb=load(vectorstore=vectorstore)
    except Exception as e:
        logger(f"An error occured trying to load the vector store: {e}", "error")
        return None

    # search vector store for documents similar to user query, return to 5 results
    kwargs={'score_threshold': score_threshold}
    try:
        docs_with_scores=vdb.similarity_search_with_relevance_scores(
            query=question,
            k=result_threshold,
            **kwargs
        )
    except Exception as e:
        logger(f"An error occured trying to search the vector store: {e}", "error")
        return None

    # if there are no results
    if len(docs_with_scores) == 0:
        logger(f"There were no documents matching your query: {question}", "error")
        return None

    # Setup llm chain
    if vectorstore["model"] in openai_inference_models:
        llm=ChatOpenAI(
            openai_api_key=api_key,
            temperature=llm_temperature,
            verbose=True,
            model=vectorstore["model"]
        )
    elif vectorstore["model"] in local_models:
        llm=LlamaCpp(
            verbose=True,
            model_path=vectorstore["path"],
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens,
            temperature=llm_temperature,
        )
    else:
        # should not happen
        logger(f"Model not found!", "error")
        return None

    return docs_with_scores, llm