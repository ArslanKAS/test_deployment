from PIL import Image
import streamlit as st
from main_functions import *
from streamlit_modal import Modal

#--------------------------------- Streamlit Code ---------------------------------

# Cover Page
image = Image.open('app_cover.png')
st.image(image, caption='Sky is the Limit')

st.header("Select Page")
table_name = st.selectbox('', ['Microsoft Azure'])

# Generate Global DataFrame 
dataframe = pd.DataFrame()

# Create Session for DataFrame so variable won't lose value
if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = dataframe


# ================================ COLUMNS ===================================
col1, col2, col3, col4 = st.columns(4)
# ================================ COLUMNS ===================================

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
        st.subheader('Insert Pinecone API Key')
        pine_api_key = st.text_input(label = "Enter Pinecone API Key", type="password", placeholder="XXXXXXXXXXXXXXXXXX")
        st.session_state["pine_api_key"] = pine_api_key
        st.session_state[openai.api_key] = api_key
        if st.button("Submit"):
            modal.close()


# Sidebar Menu
with st.sidebar:
    # Web Scrape Field and Button
    url = st.text_input("Enter the URL of the website to scrape:", placeholder="https://en.wikipedia.org")

    # Scraping Button:
    if st.button("Scrape"):
        # Scrap Data from URL
        data = scrape_sentences(url)

        # Check to Ensure the Data is not Empty
        if data is not None and not data.empty:

            # Display Data
            st.dataframe(data)
            # Add Data to the Global DataFrame
            web_dataframe = dataframe.append(data)
            # Store DataFrame in Session so variable won't lose value
            st.session_state["web_dataframe"] = web_dataframe
            # Convert DataFrame to CSV
            data_csv = web_dataframe.to_csv(index=False)
            # EncodeCSV so it can be linked
            b64 = base64.b64encode(data_csv.encode()).decode()
            # Generatea link for the Encoded CSV
            href = f'<a href="data:file/csv;base64,{b64}" download="scraped_data.csv">Download CSV File</a>'
            # Display the Download link for the Encoded CSV
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.write("No sentences found that meet the criteria.")

    # Want Embedding?
    table_name = st.text_input("Enter Link Name without Spaces")
    if st.button("Embedding"):
        # Get Data from the scraped URL
        web_dataframe = st.session_state["web_dataframe"]
        openai.api_key = st.session_state[openai.api_key]
        embed_dataframe = embedding(web_dataframe)
        st.session_state["embedd_dataframe"] = embed_dataframe

    if st.button("Data 2 Pine"):
        st.write(st.session_state)
        pinecone.init(api_key = st.session_state["pine_api_key"])
        pine_index = df2pine(st.session_state["embedd_dataframe"])
        st.session_state["pine_index"] = pine_index
        st.write(pine_index)

    if st.button("Pine 2 Data"):
        pine_dataframe = pine2df()
        st.session_state["pine_dataframe"] = pine_dataframe
 

# if col3.button("Set OpenAI API Key"):
# api_key = st.text_input(label = "Enter OpenAI API Key", type="password")
# openai.api_key = api_key
# st.session_state[openai.api_key] = api_key
    # st.write(st.session_state["api_key"])


# Search Query Field and Button
search_query = st.text_input("Enter Search Query Relevant Page:", placeholder="What is Wikipedia?")
if st.button("Show results"):
    
    openai.api_key = st.session_state[openai.api_key]
    pine_df = st.session_state["pine_dataframe"]
    show_results(pine_df, search_query)


# Models
# gpt-3.5-turbo
# text-embedding-ada-002
# To Generate Requirements File: pip list --format=freeze > requirements.txt
# To Install Requirements from File: pip install requirements.txt