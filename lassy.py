import streamlit as st
import documents as docs
import vectorstore as vs
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Import app configuration
from config import openai_inference_models, local_models

def submit(url, query, model_name, reindex):
    index = None

    if url:
        # Load documents
        index = docs.load_documents(url, model_name, reindex)
    
    if query and index is not None:
        # answer user query
        results = vs.search(query, index)
    else:
        return None

    chain = load_qa_chain(llm=results[1], chain_type="stuff")

    # Get query response based on all matches
    query_response = chain.run(input_documents=results[0], question=query)
    with st.expander("**Answer:**"):
        st.write(query_response)
    st.write("---")

    # display results
    st.header("Related Search Results")
    for document_tuple in results[0]:
        document = document_tuple[0]
        score = document_tuple[1]
        doc = [document]
        source = doc[0].metadata['source']
        try:
            title = doc[0].metadata['title']
        except KeyError:
            title = "Missing Title"
        st.write(f"{title}")
        st.write(f"**{source}** Score: {score}")

        #Callback and Query Information
        with get_openai_callback() as cb:
            quiestion = "In three sentences or less, summarize how the document relates to the following query: " + query
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
    # Create sidebar widget
    sidebar = st.sidebar.empty()

    with sidebar:
        st.title("LLM Semantic Site Search")
        st.markdown('''
        ## About
        This app is an LLM powered search engine that crawls a sitemap and performs a semantic search of the site based on your query.
        - [View the source code](https://github.com/maociao/llm-semantic-site-search)
        ''')

    # Create search form
    search_container = st.empty()

    url = ""
    reindex = False
    query = ""

    with search_container.container():
        st.header("LLM Semantic Site Search")

        form = st.form(key='search_form')
        model_list = ["model"] + openai_inference_models + local_models
        model_name = form.selectbox("Select a model",
            tuple(model_list),
            index=0
        )
        url = form.text_input("Enter the url of the site to search")
        reindex = form.checkbox("Reindex vector store?")
        query = form.text_area("Ask something about the site",
                    placeholder="Does this site contain any information about bananas?"
        )
        form.form_submit_button("Run", on_click=submit(url, query, model_name, reindex))

if __name__ == '__main__':
    main()