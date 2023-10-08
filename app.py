import streamlit as st
import uuid
import os
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
    This app is an LLM powered chat bot that crawls a website's sitemap and answers semantic search queries about the site.
    Depending on the size of the site, this could take a while if it is the first time this site was catalogued.
    - [View the source code](https://github.com/maociao/llm-semantic-site-search)
    ''')

def submit(url=str, query=str, model_name=str, overwrite=bool):
    run(url,query, model_name, overwrite)

def run(url=str, query=str, model_name=str, overwrite=bool):    
    if url:
        # check if url is domain only if not then split the url into domain and path
        if "/" in url:
            url = url.split("/")[2]

        # load our model
        if model_name in "gpt-3.5-turbo-16k":
            # check if OPENAI_API_KEY is set
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key == "":
                st.error("Error: environment variable OPENAI_API_KEY is not set")
                return()
            model_type = "openai"
            embedding = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
        elif model_name in ["ggml-alpaca-7b-q4", "mistral-7B-v0.1"]:
            curdir = os.path.curdir
            model_path = os.path.join(curdir, "models", model_name + ".bin")
            # check if model_path exists
            if not os.path.exists(model_path):
                st.error(f"Error: {model_name} model does not exist")
                return()
            model_type = "local"
            embedding = LlamaCppEmbeddings(model_path)
        else:
            model_type = "none"
            return()

        # Load the sitemap.xml file from the url
        sitemap_url = f"https://{url}/sitemap.xml"

        print(f"Loading {sitemap_url}")

        try:
            response = requests.get(sitemap_url)
        except Exception as e:
            st.error(f"Error fetching URL {sitemap_url}: {e}")
            return()
        
        sitemap_xml = response.text

        metadatas = []
        texts = []
        ids = []

        # loop through all urls in sitemap_xml
        link_list = bs(sitemap_xml, "xml")
        links = link_list.find_all("loc")
        num_links = len(links)

        if num_links == 0:
            st.error(f"Error: no links found in {sitemap_url}")
            return()

        store_name = f"{url}.vdb"
        # check for vdb overwrite
        if not os.path.exists(store_name) or overwrite:
            # load site pages
            load_status = st.progress(0, text=f"Loading sitemap")

            # limit our results to top 10 <-- DEBUG CODE
            num_links = 3

            for i, link in enumerate(links):
                link = link.string

                load_status.progress(i/num_links, text=f"Loading {link}")

                # limit our results to top 10 <-- DEBUG CODE
                if i == num_links:
                    break
                
                # get page
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

                metadata = {'source': link}
                metadatas.append(metadata)
                id = str(uuid.uuid4())
                ids.append(id)

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

            load_status.empty()
            
            if model_type != "none":
                # Check for and create Vector Store 
                vector_load = st.spinner(f"Updating Vector Store...")
                with vector_load:
                    print(f"texts: {texts} {len(texts)}")
                    print(f"metadatas: {metadatas} {len(metadatas)}")
                    print(f"ids: {ids} {len(ids)}")
                    print(f"embedding: {embedding}")
                    vector_store = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas, ids=ids)
                    try:
                        vector_store.save_local(store_name)
                    except RecursionError as e:
                        st.error(f"RecusionError trying to save vector store: {e}")
                        return()
        else:
            try:
                vector_store = FAISS.load_local(store_name, embeddings=embedding)
            except EOFError as e:
                st.error(f"An error occured reading the vector store: {e}")
                return()


    if query:
        #Accept User Queries
        print(f"query: {query}")
        docs = vector_store.similarity_search(query=query, k=5)

        print(f"docs: {docs}")
        st.header("Search Results")
        for doc in docs:
            print(f"doc: {doc}")
            #st.write(f"**{doc.metadata['source']}**")
            st.write("---")
            #Generate Responses Using LLM
            llm = ChatOpenAI(openai_api_key=api_key, temperature=0.9, verbose=True, model=model_name)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            #Callback and Query Information
            with get_openai_callback() as cb:
                quiestion = "In two sentences or less, describe how the document relates to the following query: " + query
                response = chain.run(input_documents=docs, question=quiestion)
                with st.expander("Document summary"):
                    st.write(response)
                with st.sidebar:
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
    model_name = form.selectbox("Select a model",
        ("model","gpt-3.5-turbo-16k",'ggml-alpaca-7b-q4','mistral-7B-v0.1'),
        index=0
    )
    url = form.text_input("Enter the url of the site to search")
    overwrite = form.checkbox("Overwrite")
    query = form.text_area("Ask something about the site",
                placeholder="Can you give me a short summary?"
    )
    form.form_submit_button("Run", on_click=submit(url=url, query=query, model_name=model_name, overwrite=overwrite))

if __name__ == '__main__':
    main()
