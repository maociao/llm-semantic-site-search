import streamlit as st
import pickle
import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS

with st.sidebar:
    st.title("LLaMa Semantic Site Search")
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

        # Load the sitemap.xml file from the url
        sitemap_url = f"https://{url}/sitemap.xml"
        response = requests.get(sitemap_url)
        sitemap_xml = response.text

        # loop through all urls in sitemap_xml and extract text from each using beautifulsoup
        link_list = BeautifulSoup(sitemap_xml, "xml")
        for link in link_list.find_all("loc"):

            # Extract the text from the url
            response = requests.get(link)
            page = BeautifulSoup(response.text, "html.parser")

            # Extract text from the page
            text = page.get_text()

            # Check for and create Vector Store 
            store_name = url
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
            else:
                embeddings = LlamaCppEmbeddings(model_path=f"models/{model_name}.bin")
                vector_store = FAISS.from_texts(text, embedding=embeddings, metadatas={"Link" : link})
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vector_store, f)

    if query:
        #Accept User Queries
        docs = vector_store.similarity_search(query=query)

        st.header("Search Results")
        for doc in docs:
            st.write(f"**{doc.metadata['Link']}**")
            st.write(doc.text)
            st.write("---")

def main():
    st.header("WEB LLM")

    form = st.form(key='my_form')
    model_name = form.selectbox(
        "Select a model",
        ('ggml-alpaca-7b-q4','another-model')
    )
    url = form.text_input("Enter the url of the site to search", value="https://", key="url")
    query = form.text_area(
                "Ask something about the site",
                placeholder="Can you give me a short summary?",
                key="question"
            )
    form.form_submit_button("Run", on_click=submit(url=url, query=query, model_name=model_name))

if __name__ == '__main__':
    main()
