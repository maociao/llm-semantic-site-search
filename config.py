# config.py
# This is the configuration file for the application
import os
#
# API Key for OpenAI only currently
api_key = os.getenv("OPENAI_API_KEY")
#
# OpenAI models
# Only one embedding model, but choice of inference models (afaik all use same embedding model)
openai_embedding_model = "text-embedding-ada-002"
openai_inference_models = ['gpt-3.5-turbo-16k','gpt-4-32k']
#
# LLaMa based models
# All local models must be downloaded and installed in the models folder
local_models = ['ggml-alpaca-7b-q4','mistral-7B-v0.1']
#
# LLM temperature. The higher the temperature the more creative the results
llm_temperature = 0.9

# This is the number of search results to return.
result_threshold = 10

# This is the score threshold for search results. Any result below this threshold is disregarded.
score_threshold = 0.5
