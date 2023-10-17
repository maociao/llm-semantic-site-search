import streamlit as st
import documents as docs
import vectorstore as vs
from utils import logger
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Import app configuration
from config import openai_inference_models, local_models, score_threshold, confluence_url, confluence_api_key, confluence_username

search_container=st.empty()

def submit(url: str, query: str, model_name: str, space: str, username: str, api_key: str, reindex: bool=False, attachments: bool=False):
    index=None

    if url:
        # Load documents
        index=docs.load_confluence_documents(
            source=url, 
            model=model_name,
            space_key=space,
            username=username,
            api_key=api_key,
            include_attachments=attachments,
            reindex=reindex,
        )

    if query and index is not None:
        # answer user query
        results=vs.search(query, index)
    else:
        return None

    # results should be a tuple of documents and llm definition
    if results is not None:

        # sort results by score descending highest to lowest
        sorted_docs_with_scores=sorted(
            results[0],
            key=lambda x: x[1],
            reverse=True
        )

        documents=[]
        seen_sources=set()
        unique_docs_with_scores=[]

        # filter results by scores and remove duplicate sources
        for document_tuple in sorted_docs_with_scores:
            document=document_tuple[0]
            score=document_tuple[1]
            source=document.metadata['source']

            # the FAISS kwargs score_threshold does not seem to always work
            if score < score_threshold:
                continue

            # Remove duplicate sources
            if source not in seen_sources:
                seen_sources.add(source)
                unique_docs_with_scores.append(document_tuple)
            
            # capture all docs for query_response
            documents.append(document_tuple[0])

        chain=load_qa_chain(llm=results[1], chain_type="stuff")

        # Get query response based on all matches
        try:
            query_response=chain.run(input_documents=documents, question=query)
        except Exception as e:
            st.error(f"An error occured trying to run the query: {e}")
            return None

        with st.expander("**Answer:**"):
            st.write(query_response)
        st.write("---")

        # display results
        st.header("Related Search Results")
        for document_tuple in unique_docs_with_scores:
            document=document_tuple[0]
            score=document_tuple[1]
            doc=[document]
            source=doc[0].metadata['source']
#            date=doc[0].metadata['date']
#            content_type=doc[0].metadata['content-type']
#            language=doc[0].metadata['language']
            try:
                title=doc[0].metadata['title']
            except KeyError:
                title="Missing Title"
            st.markdown(f"### {title}")
#            st.write(f"Date: {date} Content Type: {content_type} Language: {language}")
            st.write(f"**{source}**")
            st.write(f"Score: {score}")

            #Callback and Query Information
            with get_openai_callback() as cb:
                quiestion="In three sentences or less, summarize how the document relates to the following query: " + query
                response=chain.run(input_documents=doc, question=quiestion)
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
    return

def main():
    # Create sidebar widget
    sidebar=st.sidebar.empty()

    with sidebar:
        st.title("LLM Semantic Confluence Search")
        st.markdown('''
        ## About
        This app is an LLM powered search engine that crawls your confluence space and performs a semantic search of the space based on your query.
        - [View the source code](https://github.com/maociao/llm-semantic-site-search)
        ''')

    # Create search form
    search_container.empty()

    url=""
    reindex=False
    query=""
    username=""
    api_key=""
    space=""
    attachments=False
    
    with search_container.container():
        st.header("LLM Semantic Confluence Search")

        form=st.form(key='search_form')
        model_list=["model"] + openai_inference_models + local_models
        model_name=form.selectbox("Select a model",
            tuple(model_list),
            index=0
        )
        url=form.text_input("Enter the url of your Confluence space",
            placeholder="https://my-confluence-space.atlassian.net/wiki",
            value=confluence_url
        )
        space=form.text_input("Enter the space key",
            placeholder="MY-SPACE"
        )
        if confluence_username == "":
            username=form.text_input("Enter your Atlassian account username",
                placeholder="optional if confluence_username is set in config.py"
        )
        else:
            username=confluence_username
        if confluence_api_key == "":
            api_key=form.text_input("Enter your Atlassian account API key",
                placeholder="optional if confluence_username is set in config.py"
            )
        else:
            api_key=confluence_api_key
        attachments=form.checkbox("Include attachments?")
        reindex=form.checkbox("Reindex vector store?")
        query=form.text_area("What is it that you want to know?",
                    placeholder="What is this wiki about?"
        )
        logger(f"Loading documents from Confluence space {url} with username {username} and api key {api_key}", "info")

        form.form_submit_button("Run", on_click=submit(url, query, model_name, space, username, api_key, reindex, attachments))

if __name__ == '__main__':
    main()