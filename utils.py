import os
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)

replaceable=st.empty()

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

import pickle


# save data
def write_pickle(data, docstore):
    file=os.path.join(os.path.dirname(__file__), "data", docstore)
    with open(file, 'wb') as f:
        pickle.dump(data, f)

# lazy save data
def write_pickle_lazy(data, docstore):
    file=os.path.join(os.path.dirname(__file__), "data", docstore)
    with open(file, 'wb') as f:
        for item in data:
            pickle.dump(item, f)

# load data
def read_pickle(docstore):
    file=os.path.join(os.path.dirname(__file__), "data", docstore)
    with open(file, 'rb') as f:
        return pickle.load(f)

# lazy load data
def read_pickle_lazy(file):
    with open(file, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

