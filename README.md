# LLM Semantic Site Search

This project was taken on a weekend to learn about langchain and semantic search and how to build a search engine using streamlit.  

I used the https://github.com/sychhq/sych-blog-llm-qa-app.git as a template for this project.

This app is an LLM powered search engine that crawls a website's sitemap and performs a semantic search on the site, returning top results with an AI description of the relevancy of the link to your query.

### Instructions

#### Clone Repo

```
git clone https://github.com/maociao/llm-semantic-site-search.git
```

#### Create And Activate Virtual Environemnt

```
cd llm-semantic-site-search
python -m venv llm_search_app_venv
source llm_search_app_venv/bin/activate
```

#### Install Dependencies

```
pip install -r requirements.txt
```

#### Run The App

```
streamlit run app.py
```
