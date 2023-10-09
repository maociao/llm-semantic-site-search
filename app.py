import streamlit as st
import os
import requests
from bs4 import BeautifulSoup as bs
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.document_transformers import Html2TextTransformer

with st.sidebar:
    st.title("LLM Semantic Site Search")
    st.markdown('''
    ## About
    This app is an LLM powered search engine that crawls a website's sitemap and performs a semantic search on the site, returning top results with an AI description of the relevancy of the link to your query.
    - [View the source code](https://github.com/maociao/llm-semantic-site-search)
    ''')

def submit(url=str, query=str, model_name=str, overwrite=bool):
    run(url, query, model_name, overwrite)

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

        docs = []

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

            # limit our results for debugging <-- DEBUG CODE
            #num_links = 5

            for i, link in enumerate(links):
                link = link.string

                load_status.progress(i/num_links, text=f"Loading {link}")

                # limit our results for debugging <-- DEBUG CODE
                #if i == num_links:
                #   break
                
                # get page
                try:
                    response = requests.get(link)
                    response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx and 5xx)
                except requests.exceptions.SSLError as e:
                    st.error(f"skipping {link} due to SSL error: {e}")
                    continue
                except requests.exceptions.Timeout as e:
                    st.error(f"skipping {link} due to Timeout error: {e}")
                    continue
                except requests.exceptions.TooManyRedirects as e:
                    st.error(f"skipping {link} due to Too many redirects: {e}")
                    continue
                except requests.exceptions.RequestException as e:
                    st.error(f"skipping {link} due to Request failed: {e}")
                    continue
                except requests.exceptions.ConnectionError as e:
                    st.error(f"skipping {link} due to Connection error: {e}")
                    continue

                header_template = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Cache-Control": "no-cache"
                }

                # check content-type and select appropriate document loader
                content_type = response.headers.get('content-type')

                # check if content_type is None before checking for substrings
                if content_type is None:
                    st.error(f"Unable to determine content type for {link}")
                    continue 
                
                # skip to the next iteration of the loop
                if 'application/pdf' in content_type:
                    try:
                        loader = PyPDFLoader(link) 
                        docs.extend(loader.load_and_split())
                    except Exception as e:
                        st.error(f"Error loading PDF {link}: {e}")
                        continue
                elif 'text/html' in content_type:
                    loader = WebBaseLoader(link,
                                verify_ssl=False,
                                header_template=header_template
                    )
                    html2text = Html2TextTransformer()
                    try:
                        doc = loader.load_and_split()
                        docs.extend(html2text.transform_documents(doc))
                    except Exception as e:
                        st.error(f"Error loading HTML {link}: {e}")
                        continue
                elif 'text/plain' in content_type:

                else:
                    st.error('Unhandled content type skipped: {}'.format(content_type))
                    continue

            load_status.empty()
            
            # refactor this to add to vetor store in for loop
            if model_type != "none":
                # Check for and create Vector Store 
                vector_load = st.spinner(f"Updating Vector Store...")
                with vector_load:
                    vector_store = FAISS.from_documents(docs, embedding=embedding) # metadatas=metadatas, ids=ids
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


    # Run user query. check on model is to keep streamlit from running query without a model
    if query and model_name != "model":

        # search vector store for documents similar to user query, return to 5 results
        kwargs = {'score_threshold': 0.6}
        try:
            docs_with_scores = vector_store.similarity_search_with_relevance_scores(query=query, k=20, **kwargs)
            sorted_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            st.error(f"An error occured trying to search the vector store: {e}")
            return()
        
        documents = []
        seen_sources = set()
        unique_docs_with_scores = []

        # filter results by scores and remove duplicate sources
        for document_tuple in sorted_docs_with_scores:
            document = document_tuple[0]
            score = document_tuple[1]
            source = document.metadata['source']

            if score < 0.2:
                continue

            if source not in seen_sources:
                # If the source is not in seen_sources, add it to seen_sources and keep the document_tuple
                seen_sources.add(source)
                unique_docs_with_scores.append(document_tuple)
            
            # capture docs for query_response
            documents.append(document_tuple[0])

        # if there are no results
        if len(unique_docs_with_scores) == 0:
            st.write(f"There were no documents matching your query: {query}")
            return()

        # Setup llm chain
        llm = ChatOpenAI(openai_api_key=api_key, temperature=0.9, verbose=True, model=model_name)
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        # Get query response based on all matches
        query_response = chain.run(input_documents=documents, question=query)
        with st.expander("**Answer:**"):
            st.write(query_response)
        st.write("---")

        # display results
        st.header("Related Search Results")
        for document_tuple in unique_docs_with_scores:
            document = document_tuple[0]
            score = document_tuple[1]
            doc = [document]
            source = doc[0].metadata['source']
            title = doc[0].metadata['title']
            st.write(f"{title}")
            st.write(f"**{source}** Score: {score}")

            #Callback and Query Information
            with get_openai_callback() as cb:
                quiestion = "In two sentences or less, summarize how the document relates to the following query: " + query
                response = chain.run(input_documents=doc, question=quiestion)
                with st.expander("Document summary"):
                    st.write(response)
                st.write("---")
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
    overwrite = form.checkbox("Overwrite vector store?")
    query = form.text_area("Ask something about the site",
                placeholder="Can you give me a short summary?"
    )
    form.form_submit_button("Run", on_click=submit(url=url, query=query, model_name=model_name, overwrite=overwrite))

if __name__ == '__main__':
    main()
