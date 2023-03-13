import os
import csv
import base64
import requests
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper

# Web Scraping
def scrape_sentences(url):
    # Send a request to the URL and get the HTML content
    r = requests.get(url)
    html_content = r.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all the sentences in the HTML content
    sentences = []
    for p in soup.find_all("p"):
        sentences.extend(p.text.split(". "))
    for td in soup.find_all("td"):
        sentences.extend(td.text.split(". "))
    for li in soup.find_all("li"):
        sentences.extend(li.text.split(". "))

    # Filter out sentences that are less than 20 characters
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
    # Create a Pandas dataframe where each sentence is a row
    df = pd.DataFrame(sentences, columns=["Sentences"])
    web_df = df
    return web_df

# Saving DataFrame as Text File
def savedftxt(web_df, url_title):
    # create directory with url_title if it doesn't exist
    dir_path = f"contents/{url_title}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # save dataframe as text file in the created directory
    file_path = f"{dir_path}/{url_title}.txt"
    web_df.to_csv(file_path, sep="\n", index=False, header=None)
    # web_df.to_csv(f"contents\{url_title}\{url_title}.txt", sep="\n", index=False, header=None)

# Embedding Function
def embedding(url_title):
    dir_path = f"contents/{url_title}"
    # with open(path, "r") as f:
    #     file = f.read()

    # set maximum input size
    max_input_size = 4096 #4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 #600

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, model_name="text-ada-001", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(dir_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    filename = url_title.rstrip(".txt")
    index.save_to_disk(f'indexes/{filename}.json')

def ask_results(filename, search_query):
    file_path = f"indexes/{filename}"
    index = GPTSimpleVectorIndex.load_from_disk(file_path)
    # while True: 
    response = index.query(search_query, response_mode="compact")
    st.write(response.response, unsafe_allow_html=True)