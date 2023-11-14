# config.py
# This is the configuration file for the application
import os
#
### OpenAI models ###
#
# API Key for OpenAI only currently
api_key=os.getenv("OPENAI_API_KEY")
#
# Only one embedding model, but choice of inference models (afaik all use same embedding model)
openai_embedding_model="text-embedding-ada-002"
openai_inference_models=['gpt-3.5-turbo-16k','gpt-4','gpt-4-32k-0613','gpt-4-32k']
#
# LLM temperature. The higher the temperature the more creative the results
llm_temperature=0.9
#
# This is the number of search results to return.
result_threshold=10
#
# This is the score threshold for search results. Any result below this threshold is disregarded.
score_threshold=0.5
#
### LLaMa based models ###
#
# All local models must be downloaded and installed in the models folder
models_dir=os.path.join(os.path.dirname(__file__), "models")
# create a list of all filenames from models dir
local_models=[os.path.splitext(file)[0] for file in os.listdir(models_dir) if file.endswith(".gguf")]
#
# model context window
n_ctx=4096
#
# max number of tokens for the model to return
max_tokens=1024
#
# nomber of layers to load in gpu [Requires CUDA]
n_gpu_layers=32
#
### Confluence ###
#
confluence_url="https://matt-oberpriller.atlassian.net/wiki"
#
confluence_api_key=os.getenv("CONFLUENCE_API_KEY")
#
confluence_username="matt.oberpriller@gmail.com"
#
tesseract_path="D:\Program Files\Tesseract-OCR\tesseract.exe"
