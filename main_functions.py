import csv
import base64
import openai
import swifter
import sqlite3
import pinecone
import requests
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

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

# Embedding Function
def embedding(web_df):
        # Get Embeddings related to the Sentences
        web_df['Embedding'] = web_df['Sentences'].map(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        # Convert Embeddings to Numpy Arrays
        web_df['Embedding'] = web_df['Embedding'].swifter.apply(np.array)
        emb_df = web_df
        return emb_df

# Similarity Function
def similarity(df, search_query):
    # Get Embedding related to the Search Query
    search_term_vector = get_embedding(search_query, engine='text-embedding-ada-002')
    # Generate Similarities related to Searched Embedding

    # st.write(type(search_term_vector))
    df['Similarities'] = df['Embedding'].swifter.apply(lambda x: cosine_similarity(x, search_term_vector))
    return df

def df2pine(emb_df):
    # Connect with Index
    index_name = "scrapetoai"
    index = pinecone.Index(index_name)

    # Upsert the Data on Pinecone into Batches
    from tqdm.auto import tqdm

    count = 0  # we'll use the count to create unique IDs
    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(emb_df['Sentences']), batch_size)):
        # set end position of batch
        i_end = min(i+batch_size, len(emb_df['Sentences']))
        # get batch of lines and IDs
        lines_batch = emb_df['Sentences'][i: i+batch_size]
        ids_batch = [str(n) for n in range(i, i_end)]
        # prep metadata and upsert batch
        meta = [{'text': line} for line in lines_batch]
        emb = emb_df['Embedding'].apply(lambda x: x.tolist())
        to_upsert = zip(ids_batch, emb, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))

    return index

def pine2df():

    # Connect with Index
    index_name = "scrapetoai"
    index = pinecone.Index(index_name)
    data_length = index.describe_index_stats()["total_vector_count"]

    # Fetch all the results
    fetch_response = index.fetch(ids=[str(x) for x in list(range(0, data_length))])

    # Create an Empty DataFrame
    pine_df = pd.DataFrame(columns=["Sentences", "Embedding"])

    # Put all the fetched data into the DataFrame
    for i in range(data_length):
        pine_df = pine_df.append({
            "Sentences": fetch_response.get("vectors")[str(i)]["metadata"]["text"], 
            "Embedding": fetch_response.get("vectors")[str(i)]["values"]}, 
            ignore_index=True)

    return pine_df

# Show Final Results
def show_results(pine_df, search_query):
    simi_df = similarity(pine_df, search_query)
    if "Embedding" in simi_df.columns:
        if any(simi_df["Similarities"] > 0.8):
            final_df = simi_df[simi_df["Similarities"] > 0.8].sort_values('Similarities', ascending=False)
            results = ' '.join(final_df['Sentences'].tolist())
            st.write(results)
        else:
            st.write("Sorry I don't have the relevant information")
    else:
        pass