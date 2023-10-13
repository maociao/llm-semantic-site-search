import os
import uuid
import utils
import logging
import requests
import datetime, pytz
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

class DocumentLoader():
    def __init__(self):
        self.documents = []
        self.metadata = {}
        self.metadata['source'] = ''
        self.metadata['title'] = ''
        self.metadata['author'] = ''
        self.metadata['date'] = ''
        self.metadata['keywords'] = ''
        self.metadata['abstract'] = ''
        self.metadata['language'] = ''
        self.metadata['type'] = ''
        self.metadata['format'] = ''
        self.metadata['identifier'] = ''
        self.metadata['coverage'] = ''
        self.metadata['rights'] = ''
        self.metadata['relation'] = ''
        self.metadata['size'] = ''
        self.metadata['format'] = ''
        self.vectorstore = ''

    def get_documents(self):
        return self.documents
    
    def get_document(self, index):
        return self.documents[index]

    def get_document_count(self):
        return len(self.documents)

    def get_metadata(self):
        return self.metadata

    def load_documents(self, source):
        self.documents = []

        # source is a file path, check if it's a directory or a file
        if os.path.isdir(source):
            # source is a directory, load documents from directory
            for filename in os.listdir(source):
                self.load_documents(os.path.join(source, filename))

        # load document from file
        if os.path.isfile(source):

            loader = ''

            # get filename
            filename = os.path.basename(source)

            # set some metadata
            self.metadata['identifier'] = uuid.uuid5(uuid.NAMESPACE_URL, source)
            self.metadata['source'] = source
            self.metadata['size'] = os.path.getsize(os.path.join(source, filename))

            #current date and time
            current_datetime = datetime.datetime.now(pytz.utc)
            self.metadata['date'] = current_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")
            
            # set document type and format and load document
            if filename.endswith(".txt"):
                self.metadata['format'] = 'text/plain'
                self.metadata['type'] = 'text'
                loader = TextLoader(
                    file_path=os.path.join(source, filename),
                )
            if filename.endswith(".docx") or filename.endswith(".doc"):
                self.metadata['format'] = 'application/msword'
                self.metadata['type'] = 'MS Word document'
                loader = UnstructuredWordDocumentLoader(
                    file_path=os.path.join(source, filename)
                )
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                self.metadata['format'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                self.metadata['type'] = 'MS Excel document'
                loader = UnstructuredExcelLoader(
                    file_path=os.path.join(source, filename)
                )
            if filename.endswith(".pdf"):
                self.metadata['format'] = 'application/pdf'
                self.metadata['type'] = 'PDF document'
                loader = PyPDFLoader(
                    file_path=os.path.join(source, filename)
                )

            # unsupported file type
            if loader == '':
                raise Exception(f"Unsupported file type: {filename}")

            # enrich data with llm prompt
            self.metadata['title'] = ''
            self.metadata['author'] = ''
            self.metadata['date'] = ''
            self.metadata['keywords'] = ''
            self.metadata['abstract'] = ''
            self.metadata['language'] = ''

            # load documents
            self.documents = loader.load_and_split()

            # save document embeddings
            self.vectorstore = vs.get_vectorstore()
            vs.save(self.documents, self.vectorstore)

        # source is a URL, load documents from web
        elif source != '':

            # split url into domain, path, and query
            if "/" in source:
                parsed = urlparse(source)
                url = parsed.netloc
            else:
                url = source

            vectorestore_name = url

            # Load the sitemap.xml file from the url
            sitemap_url = f"https://{url}/sitemap.xml"
            try:
                logging.info(f"Fetching {sitemap_url}")
                response = requests.get(sitemap_url)
            except Exception as e:
                raise Exception(f"Error fetching URL {sitemap_url}: {e}")
            
            sitemap_xml = response.text
            install_dir = os.path.dirname(os.path.abspath(__file__))

            # read local sitemap.xml override for testing
            test_file = os.path.join(install_dir, 'test_sitemap.xml')
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    sitemap_xml = f.read()

            # load sitemap links
            link_list = bs(sitemap_xml, "xml")
            links = link_list.find_all("loc")
            num_links = len(links)

            if num_links == 0:
                raise Exception(f"Error: no links found in {sitemap_url}")

            logging.info(f"Found {num_links} links in {sitemap_url}")

            progress = 0

            # load documents from links
            for i, link in enumerate(links):
                link = link.string
                loader = ''
                docs = []

                # set some metadata
                self.metadata['identifier'] = uuid.uuid5(uuid.NAMESPACE_URL, source)
                self.metadata['source'] = source
                self.metadata['size'] = ''

                #current date and time
                current_datetime = datetime.datetime.now(pytz.utc)
                self.metadata['date'] = current_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")

                # update progress
                if i == 0:
                    self.update_progress(1)       
                else:             
                    progress = i / len(links)
                    self.update_progress(progress)

                logging.info(f"Fetching {link}")

                # get page
                try:
                    response = requests.get(link)
                    response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx and 5xx)
                except requests.exceptions.RequestException as e:
                    raise Warning(f"skipping {link} due to Request failed: {e}")

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
                    raise Warning(f"Unable to determine content type for {link}")
                
                # Load and parse document based on document type
                if 'application/pdf' in content_type:
                    loader = PyPDFLoader(
                        file_path=link,
                        headers=header_template
                    )

                    logging.info(f"Loading PDF {link}")

                    try:
                        pdf_docs = loader.load_and_split()
                        # the pdf loader replaces the link as the source with a local filename.
                        # this is to restore the original link as the docuemnt source (Issue: #3)
                        for i in enumerate(pdf_docs):
                            pdf_docs[i[0]].metadata["source"] = link
                    except Exception as e:
                        raise Warning(f"Failed loading PDF {link}: {e}")
                    
                    docs = pdf_docs

                if 'text/html' in content_type:
                    loader = WebBaseLoader(
                        link,
#                        verify_ssl=False,
                        header_template=header_template
                    )

                    logging.info(f"Loading HTML {link}")

                    html2text = Html2TextTransformer()
                    try:
                        html_docs = loader.load_and_split()
                        docs.extend(html2text.transform_documents(html_docs))
                    except Exception as e:
                        raise Warning(f"Error loading HTML {link}: {e}")

                # unsupported content type
                if loader == '':
                    raise Warning(f"Skipping {link} due to unsupported content type: {content_type}")

                # enrich data with llm prompt
                self.metadata['title'] = ''
                self.metadata['author'] = ''
                self.metadata['date'] = ''
                self.metadata['keywords'] = ''
                self.metadata['abstract'] = ''
                self.metadata['language'] = ''

                # save document embeddings
                logging.info(f"Saving document embeddings")
                try:
                    vs.save(self.documents, vectorestore_name)
                except Exception as e:
                    raise Exception(f"Error saving document embedding: {e}")

    def update_progress(self, progress):
        # Call back to update progress
        utils.update_progress(progress)
