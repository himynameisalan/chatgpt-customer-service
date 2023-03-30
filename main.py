from pathlib import Path

# pip install llama_index
# pip install langchain
from llama_index import download_loader, GPTSimpleVectorIndex

import os

def chat():
    # configure your api key
    os.environ['OPENAI_API_KEY'] = 'your api key'
    SimpleCSVReader = download_loader('SimpleCSVReader')
    loader = SimpleCSVReader()
    # prepare and load your q&a dataset
    documents = loader.load_data(file=Path('./raw_data.csv'))
    # transform your data
    index = GPTSimpleVectorIndex(documents)
    # enter your question and wait for the response
    response = index.query('your question')
    print(response)

if __name__ == '__main__':
    chat()
