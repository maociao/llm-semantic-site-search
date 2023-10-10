# config.py
# This is the configuration file for the application
import os
#
### OpenAI models ###
#
# API Key for OpenAI only currently
api_key = os.getenv("OPENAI_API_KEY")
#
# Only one embedding model, but choice of inference models (afaik all use same embedding model)
openai_embedding_model = "text-embedding-ada-002"
openai_inference_models = ['gpt-3.5-turbo-16k','gpt-4-32k']
#
# LLM temperature. The higher the temperature the more creative the results
llm_temperature = 0.9
#
# This is the number of search results to return.
result_threshold = 10
#
# This is the score threshold for search results. Any result below this threshold is disregarded.
score_threshold = 0.5
#
### LLaMa based models ###
#
# All local models must be downloaded and installed in the models folder
local_models = ['llama-2-7b.Q4_K_M','mistral-7b-v0.1.Q4_K_M']
#
# model context window
n_ctx = 4096
#
# nomber of layers to load in gpu [Requires CUDA]
n_gpu_layers = 32



