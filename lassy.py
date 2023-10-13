import streamlit as st
import documents
import logging

# Import app configuration
from config import api_key, result_threshold, score_threshold, openai_embedding_model, openai_inference_models, local_models, llm_temperature, n_ctx, n_gpu_layers

# Create sidebar widget
with st.sidebar:
    st.title("LLM Semantic Site Search")
    st.markdown('''
    ## About
    This app is an LLM powered search engine that crawls a website's sitemap and performs a semantic search of the site based on your query.
    - [View the source code](https://github.com/maociao/llm-semantic-site-search)
    ''')

load_progress = None

def submit(url, query, model_name, overwrite):    
    global load_progress
    if url:
        load_progress = st.progress(0)
        loader = documents.DocumentLoader()
        # Load documents
        loader.load_documents(url)
        load_progress.empty()

# Callback to update progress bar
def update_progress_bar(progress):
    global load_progress
    if load_progress != None:
        load_progress.progress(progress)

def main():
    st.header("LLM Semantic Site Search")

    url = ""
    overwrite = False
    query = ""

    form = st.form(key='my_form')
    model_list = ["model"] + openai_inference_models + local_models
    model_name = form.selectbox("Select a model",
        tuple(model_list),
        index=0
    )
    url = form.text_input("Enter the url of the site to search")
    overwrite = form.checkbox("Overwrite vector store?")
    query = form.text_area("Ask something about the site",
                placeholder="Does this site contain any information about bananas?"
    )
    form.form_submit_button("Run", on_click=submit(url, query, model_name, overwrite))

if __name__ == '__main__':
    main()