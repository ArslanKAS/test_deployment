import base64
import openai
import swifter
import requests
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from bs4 import BeautifulSoup
from streamlit_modal import Modal
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

#--------------------- Web Scraping Code --------------------

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
    return df

#--------------------------------- Streamlit Code ---------------------------------

# Cover Page
image = Image.open('app_cover.png')
st.image(image, caption='Sky is the Limit')

# Web Scrape Field and Button
url = st.text_input("Enter the URL of the website to scrape:", placeholder="https://en.wikipedia.org")

# Generate Global DataFrame 
dataframe = pd.DataFrame()
# Create Session for DataFrame to variable won't lose value
if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = dataframe

# Columns to Accomdate Three Buttons
col1, col2, col3, col4 = st.columns(4)

# if url:
if col1.button("Scrape"):
    # Get Data from the scraped URL
    data = scrape_sentences(url)

    # Check to Ensure the Data is not Empty
    if data is not None and not data.empty:

        # Display Data
        st.dataframe(data)

        # Add Data to the Global DataFrame
        dataframe = dataframe.append(data)

        # Store DataFrame in Session so variable won't lose value
        st.session_state["dataframe"] = dataframe

        # Convert DataFrame to CSV
        data_csv = dataframe.to_csv(index=False)

        # EncodeCSV so it can be linked
        b64 = base64.b64encode(data_csv.encode()).decode()

        # Generatea link for the Encoded CSV
        href = f'<a href="data:file/csv;base64,{b64}" download="scraped_data.csv">Download CSV File</a>'

        # Display the Download link for the Encoded CSV
        col3.markdown(href, unsafe_allow_html=True)

    else:
        col3.write("No sentences found that meet the criteria.")


# Pop-Up for OpenAI API Key
modal = Modal(key="Demo Modal", title="")
open_modal = col4.button("OpenAI API Key")
if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        st.subheader('Insert OpenAI API Key')
        api_key = st.text_input(label = "Enter OpenAI API Key", type="password", placeholder="XXXXXXXXXXXXXXXXXX")
        openai.api_key = api_key
        st.session_state[openai.api_key] = api_key
        if st.button("Submit"):
            modal.close()


# if col3.button("Set OpenAI API Key"):
# api_key = st.text_input(label = "Enter OpenAI API Key", type="password")
# openai.api_key = api_key
# st.session_state[openai.api_key] = api_key
    # st.write(st.session_state["api_key"])


#------------------------- OpenAI Embedding Code -------------------------------

# Search Query Field and Button
search_query = st.text_input("Enter your search query relevant to URL data:", placeholder="What is Wikipedia?")
if st.button("Show results"):

    df = st.session_state["dataframe"]
    openai.api_key = st.session_state[openai.api_key]

    # Get Embeddings related to the Sentences
    df['Embedding'] = df['Sentences'].map(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

    # Convert Embeddings to Numpy Arrays
    df['Embedding'] = df['Embedding'].swifter.apply(np.array)
 
    # Get Embedding related to the Search Query
    search_term_vector = get_embedding(search_query, engine='text-embedding-ada-002')

    # Generate Similarities related to Searched Embedding
    df['Similarities'] = df['Embedding'].swifter.apply(lambda x: cosine_similarity(x, search_term_vector))

    # Sort Similarities
    df = df.sort_values('Similarities', ascending=False).head(10)

    # Final Results
    results = ' '.join(df['Sentences'].tolist())
    # results = df['Sentences'].str.join("").sum()
    st.markdown(f"**{results}**")


# To Generate Requirements File: pip list --format=freeze > requirements.txt
# To Install Requirements from File: pip install requirements.txt