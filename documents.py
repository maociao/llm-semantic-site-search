import os
import ast
import time
import uuid
import logging
import requests
import datetime, pytz
import streamlit as st
import vectorstore as vs
from urllib.parse import urlparse
from bs4 import BeautifulSoup as bs
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_transformers import Html2TextTransformer

logging.basicConfig(level=logging.INFO)

documents = []
metadata = {}
metadata['title'] = ''
metadata['author'] = ''
metadata['keywords'] = ''
metadata['abstract'] = ''
metadata['language'] = ''
metadata['format'] = ''
metadata['coverage'] = ''
metadata['rights'] = ''
metadata['relation'] = ''
metadata['size'] = ''
vectorstore = {}

status_bar = st.empty()
replaceable = st.empty()

def logger(message, type):
    global replaceable
    if type == "warning":
        logging.warning(message)
        replaceable.warning(message)
    elif type == "error":
        logging.error(message)
        replaceable.error(message)
    else:
        logging.info(message)
    return None

def load_documents(source, model, reindex):
    global replaceable

    logging.info(f"Calling load_documents({source}, {model}, {reindex})")

    documents = []

    # source is a URL, load documents from web
    if source != '' and model != "model":

        # split url into domain, path, and query
        if "/" in source:
            parsed = urlparse(source)
            url = parsed.netloc
        else:
            url = source

        # get vectorstore
        vectorstore = vs.get_vectorstore(url, model)
        if vectorstore is None:
            logger(f"Error: vectorstore not found for {source}", "error")
            return None

        # check if vectorstore exists and return it if it does
        if os.path.exists(vectorstore['path']) and not reindex:
            try:
                return vectorstore
            except Exception as e:
                logger(f"An error occured reading the vectorstore: {e}", "error")
                return None
        elif reindex:
            logger(f"Reindexing vectorstore {vectorstore['path']}", "info")

        # configure our loading status widget
        status_bar.progress(0, text=f"Loading sitemap")

        app_dir = os.path.dirname(os.path.abspath(__file__))

        # check for local sitemap.xml override
        sitemap_url = os.path.join(app_dir, 'sitemap.xml')
        if os.path.exists(sitemap_url):
            with open(sitemap_url, 'r') as f:
                sitemap_xml = f.read()

        else:
            # Load the sitemap.xml file from the url
            sitemap_url = f"https://{url}/sitemap.xml"
            logger(f"Fetching {sitemap_url}", "info")
            try:
                response = requests.get(sitemap_url)
            except Exception as e:
                logger(f"Error fetching URL {sitemap_url}: {e}", "error")
                return None
            sitemap_xml = response.text

        # load sitemap links
        soup = bs(sitemap_xml, "xml")
        loc_tags = soup.find_all('loc')
        # convert bs resultset into a list of links
        links = [tag.text for tag in loc_tags]

        # check for checkpoint register
        if os.path.exists(f".{url}.list"):
            with open(f".{url}.list", 'r') as f:
                s_links = f.read()
                links=ast.literal_eval(s_links)
        else:
            # create checkpoint register
            with open(f".{url}.list", "w") as f:
                f.write(str(links))        

        num_links = len(links)
        if num_links == 0:
            logger(f"Error: no links found in {sitemap_url}", "error")
            return None

        logger(f"Found {num_links} links in {sitemap_url}", "info")

        # load documents from links
        for index, link in enumerate(links):
            docs = []
            loader = ''

            # update progress
            status_bar.progress(index/num_links, text=f"Loading {index+1} of {num_links} pages: {link}")

            logger(f"Fetching {link}", "info")

            # get page
            content_type = None
            try:
                response = requests.head(link, allow_redirects=True)
                response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx and 5xx)
                content_type = response.headers.get('content-type')
            except requests.exceptions.RequestException as e:
                logger(f"skipping {link} due to Request failed: {e}", "warning")
                continue
            except requests.exceptions as e:
                logger(f"skipping {link} due to Request failed: {e}", "warning")
                continue

            header_template = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Cache-Control": "no-cache"
            }

            # check if content_type is None before checking for substrings
            if content_type is None:
                logger(f"Unable to determine content type for {link}", "warning")
                continue

            # Load and parse document based on document type
            if 'application/pdf' in content_type:
                logger(f"Loading PDF {link}", "info")

                try:
                    loader = PyPDFLoader(
                        file_path=link,
                        headers=header_template
                    )
                    pdf_docs = loader.load_and_split()

                    # the pdf loader replaces the link as the source with a local filename.
                    # this is to restore the original link as the docuemnt source (Issue: #3)
                    for i in enumerate(pdf_docs):
                        pdf_docs[i[0]].metadata["source"] = response.url
                except Exception as e:
                    logger(f"Failed loading PDF {link}: {e}", "warning")
                
                docs.extend(pdf_docs)

            if 'text/html' in content_type:
                loader = WebBaseLoader(
                    link,
#                        verify_ssl=False,
                    header_template=header_template
                )

                logger(f"Loading HTML {link}", "info")

                html2text = Html2TextTransformer()
                try:
                    html_docs = loader.load_and_split()
                except Exception as e:
                    replaceable.warning(f"Error loading HTML {link}: {e}")

                docs.extend(html2text.transform_documents(html_docs))

            # unsupported content type
            if loader == '':
                logger(f"Skipping {link} due to unsupported content type: {content_type}", "warning")
                continue



            # set some metadata
#            metadata['identifier'] = uuid.uuid5(uuid.NAMESPACE_URL, url)
            metadata['date'] = response.headers.get('date')
            metadata['content_type'] = content_type
            metadata['language'] = response.headers.get('content-language')
            metadata['keywords'] = ''
            metadata['abstract'] = ''
            for i in enumerate(docs):
                docs[i[0]].metadata['date'] = response.headers.get('date')
                docs[i[0]].metadata['content-type'] = content_type
                docs[i[0]].metadata['language'] = response.headers.get('content-language')
                docs[i[0]].metadata['keywords'] = ''
                docs[i[0]].metadata['questions'] = ''
                docs[i[0]].metadata['abstract'] = ''

            documents.extend(docs)

            if len(documents) == 0:
                logger(f"No documents found in {sitemap_url}", "warning")
                return None

            # save document embeddings
            logger(f"Saving document embeddings", "info")
            try:
                vs.save(documents, vectorstore)
            except Exception as e:
                logger(f"Error saving document embedding: {e}", "error")
                return None

            # update checkpoint register
            register = links[index+1:]
            if len(register) == 0:
                os.remove(f".{url}.list")
            else:
                with open(f".{url}.list", "w") as f:
                    f.write(str(register))

        # clar progress bar and messages
        time.sleep(3)
        status_bar.empty()
        replaceable.empty()
    
        return vectorstore
    
    else:
        replaceable.error(f"Error: Invalid or missing URL: {source}. Please enter a valid URL to search.")
        return None