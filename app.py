import streamlit as st
import pickle
import os
import sys
import requests
from bs4 import BeautifulSoup as bs
from PyPDF2 import PdfReader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("LLM Semantic Site Search")
    st.markdown('''
    ## About
    This app is an LLM powered chat bot that takes a website URL and answers semantic search queries about the site.
    Depending on the size of the site, this could take a while if it is the first time this site was catalogued.
    - [View the source code](https://github.com/maociao/website-llm-semantic-search)
    ''')

def submit (url, query, model_name):

    if url:
        # check if url is domain only if not then split the url into domain and path
        if "/" in url:
            url = url.split("/")[2]

        # load our model
        if model_name in "gpt-3.5-turbo-0613":
            model_type = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key == "":
                print(f"Error: environment variable OPENAI_API_KEY is not set")
                sys.exit(1)
        elif model_name in "ggml-alpaca-7b-q4":
            # load local model
            model_type = "local"
            curdir = os.path.curdir
            model_path = os.path.join(curdir, "models", model_name + ".bin")
            # check if model_path exists
            if not os.path.exists(model_path):
                print(f"Error: {model_path} does not exist") 
                sys.exit(1)
        else:
            model_type = "none"
            return()

        # Load the sitemap.xml file from the url
        sitemap_url = f"https://{url}/sitemap.xml"

        try:
            response = requests.get(sitemap_url)
        except Exception as e:
            st.error(f"Error fetching URL {sitemap_url}: {e}")
            return()

        sitemap_xml = response.text

        metadatas = []
        texts = []

        # loop through all urls in sitemap_xml
        link_list = bs(sitemap_xml, "xml")

        # limit our results to top 10 <-- DEBUG CODE
        for i, link in enumerate(link_list.find_all("loc")):
            link = link.string
            if i == 10:
                sys.exit(0)
            
            print(f"Loading {link}")
            response = requests.get(link)
            content = response.text
            # check content-type and select appropriate loader
            content_type = response.headers.get('content-type')
            if 'application/pdf' in content_type:
                parser = 'pdf' 
            elif 'text/html' in content_type:
                parser = 'html'
            else:
                st.error('Unhandled content type: {}'.format(content_type))

            metadata = {'source': link, 'content_type': content_type}
            metadatas.append(metadata)

            # load html content
            if parser == "html":
                page = bs(content, "html.parser")
                text = ""
                for string in page.stripped_strings:
                    text += string
                texts.append(text)

            # Pdf Text Extraction
            if parser == "pdf":
                pdf_reader = PdfReader(content)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                    texts.append(text)
                    metadatas.append(metadata)

        sys.exit(0)

        if model_type == "openai":
            embedding = OpenAIEmbeddings(openai_api_key=api_key)
        else:
            embedding = LlamaCppEmbeddings(model_path=model_path, n_ctx=2048)

        # Check for and create Vector Store 
        store_name = url
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            vector_store = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

    if query:
        #Accept User Queries
        docs = vector_store.similarity_search(query=query)

        st.header("Search Results")
        for doc in docs:
            st.write(f"**{doc.metadata['source']}**")
            st.write(doc.text)
            st.write("---")
            #Generate Responses Using LLM
            llm = ChatOpenAI(openai_api_key=api_key, temperature=0.9, verbose=True, model=model_name)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            #Callback and Query Information
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write("AI Response")
                st.write(response)
                st.info(f'''
                    #### Query Information
                    Successful Requests: {cb.successful_requests}\n
                    Total Cost (USD): {cb.total_cost}\n
                    Tokens Used: {cb.total_tokens}\n
                    - Prompt Tokens: {cb.prompt_tokens}\n
                    - Completion Tokens: {cb.completion_tokens}\n 
                ''')

def main():
    st.header("LLM Semantic Site Search")

    form = st.form(key='my_form')
    model_name = form.selectbox(
        "Select a model",
        ("model","gpt-3.5-turbo-0613",'ggml-alpaca-7b-q4')
    )
    url = form.text_input("Enter the url of the site to search", value="https://yoursite.com", key="url")
    query = form.text_area(
                "Ask something about the site",
                placeholder="Can you give me a short summary?",
                key="question"
            )
    form.form_submit_button("Run", on_click=submit(url=url, query=query, model_name=model_name))

if __name__ == '__main__':
    main()
