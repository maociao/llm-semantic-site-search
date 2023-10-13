from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings

class Vectorstore:
    def __init__(self) -> None:
        pass

    def save(self, documents, vectorstore_name):
        pass

    def get_vectorstore(self, name):
        return(name)