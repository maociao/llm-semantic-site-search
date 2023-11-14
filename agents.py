from langchain.chains.question_answering import load_qa_chain
from langchain import tools
from langchain.agents.agent import Agent, AgentAction, AgentFinish

output_schema = {
    "text": str, 
    "entities": [str],
    "keywords": [str],
    "summary": str
}

def get_keywords(text):
    # Extract keywords from text
    return []

def get_questions(text):
    # Extract questions from text
    return []

def get_abstract(text):
    # Generate abstract from text
    return ""


def parse(text):
    # Extract entities, keywords, generate summary, etc

    return {
        "text": text,
        "entities": [], 
        "keywords": [],
        "summary": ""  
    }

agent = Agent(tools, functions=[output_schema])

def parse(output):
    if "output_schema" in output.function_call:
        return AgentFinish(return_values=parse(output.text), log=output.text)
    else:
        # Normal agent execution
        return AgentAction(...)