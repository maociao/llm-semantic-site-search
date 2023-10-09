# LLM Semantic Site Search

This project was taken on to learn about LangChain and semantic search and how to build a search engine using Streamlit.  

Rohan Chopra's blog post, [How to augment LLMs like ChatGPT with your own data](https://sych.io/blog/how-to-augment-chatgpt-with-your-own-data/) was the inspiration for this application. I used his code, https://github.com/sychhq/sych-blog-llm-qa-app.git, as the starting point. 

This app is an LLM powered search engine that crawls a website's sitemap and performs a semantic search on the site, returning top results with an AI description of the relevancy of the link to your query.

### Technical information
Why? Well, because it was what 

* [Streamlit](https://streamlit.io/) was used as the application framework.
* For OpenAI embeddings I'm using the text-embedding-ada-002-v2 model and for the search inference I'm using the gpt-3.5-turbo-16k-0613 model.  Both were chosen based on their large context windows of 8K and 16K tokens respectively.  I went with the gpt-3 model instead of gpt-4 due to speed and cost.  More on OpenAI embeddings can be found here - [New and improved embedding model](https://openai.com/blog/new-and-improved-embedding-model)
* The vector store and search is using Meta's [FAISS](https://github.com/facebookresearch/faiss)
* The LLM workflow framework is using [LangChain](https://docs.langchain.com/docs/)

### Some learnings

* GPT programming assist is amazing at helping to write and debug code. I was able to write this and debug it over a weekend. I am a beginner python programmer with no previous experience with Streamlit or LangChain.
* LangChain has an amazing library of loaders!!! I already have ideas for more projects just from reviewing the list.

### What's next?

- [ ] Adding support for LLaMa2 and Mistral models
- [ ] Adding the choice to use an existing vector store (already uses vector store if it exists unless you choose to override)
- [x] Add a config file
- [ ] Refactor code to simplify

## Instructions

Python 3.10.11+ is required to run this app. [Download it here](https://www.python.org/downloads/)

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
