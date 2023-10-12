# LLM Semantic Site Search

This project was taken on to learn about LangChain and semantic search and how to build a search engine using Streamlit.  

Rohan Chopra's blog post, [How to augment LLMs like ChatGPT with your own data](https://sych.io/blog/how-to-augment-chatgpt-with-your-own-data/) was the inspiration for this application. I used his code, https://github.com/sychhq/sych-blog-llm-qa-app.git, as the starting point. 

This app is an LLM powered search engine that crawls a website's sitemap and performs a semantic search on the site, returning top results with an AI description of the relevancy of the link to your query.

### Technical details

* [Streamlit](https://streamlit.io/) was used for the application UI.
* [New and improved embedding model](https://openai.com/blog/new-and-improved-embedding-model) text-embedding-ada-002-v2 from OpenAI.
* [FAISS](https://github.com/facebookresearch/faiss) for vector store and similarity search.
* [LangChain](https://docs.LangChain.com/docs/) is the llm application framework

### Some learnings

* GPT programming assist is amazing at helping to write and debug code. I was able to write this and debug it over a weekend. I am a beginner python programmer with no previous experience with Streamlit or LangChain.
* LangChain has an amazing library of loaders!!! I already have ideas for more projects just from reviewing the list.

### What's next?

- [X] Add support for LLaMa2 and Mistral models
- [ ] Add the choice to use an existing vector store (already uses vector store if it exists unless you choose to override)
- [x] Add a config file
- [ ] Refactor to simplify code
- [ ] Add caching to improve performance
- [ ] Switch to using LangChain sitemap API
- [ ] Enrich vector embeddings with additional metadata
- [ ] Enrich vector embeddings with questions
- [ ] Add support for [Figma](https://python.langchain.com/docs/integrations/document_loaders/figma)
- [ ] Add support for [GitHub Issues](https://python.langchain.com/docs/integrations/document_loaders/github)
- [ ] Add support for [Confluence](https://python.langchain.com/docs/integrations/document_loaders/confluence)

## Installation

### Requirements

Python 3.10.11 - 3.11.0 is recommened to run this app. [Download it here](https://www.python.org/downloads/)
Several of the dependencies do not have wheels for Python 3.12, so you will need to install them from source if you choose to use Python 3.12.

#### Specific instructions for Windows installations

The llama-cpp-python library requires a C++ compiler.
If you want to run local models you will need to install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
If you do not plan on running local models you can skip that and just comment out the llama-cpp-python library in requirements.txt.
The app will run with just the OpenAI models.

### Clone Repo

```shell
git clone https://github.com/maociao/llm-semantic-site-search.git
```

### Create And Activate Virtual Environemnt

Linux/MacOS:
```shell
cd llm-semantic-site-search
python -m venv llm_search_app_venv
source llm_search_app_venv/bin/activate
```

Windows:
```shell
cd llm-semantic-site-search
python -m venv llm_search_app_venv
llm_search_app_venv\Scripts\activate.bat
```

### Install Dependencies

```shell
pip install -r requirements.txt
```

### Configure The App

Linux/MacOS:
```shell
export OPENAI_API_KEY="Your API key from https://platform.openai.com/"
```

Windows:
```shell
SET OPENAI_API_KEY="Your API key from https://platform.openai.com/"
```

### Run The App

```bash
streamlit run app.py
```
