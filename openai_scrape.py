from PIL import Image
import streamlit as st
from main_functions import *
from streamlit_modal import Modal

#--------------------------------- Streamlit Code ---------------------------------

# Cover Page
image = Image.open('app_cover.png')
st.image(image, caption='Sky is the Limit')

st.subheader("Select Page")
json_files = [f for f in os.listdir("indexes/") if f.endswith(".json")]
select_url_title = st.selectbox('JSON Files Appear here', json_files, label_visibility='collapsed')
# Generate Global DataFrame 
dataframe = pd.DataFrame()

# Create Session for DataFrame so variable won't lose value
if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = dataframe


# Pop-Up for OpenAI API Key
modal = Modal(key="Demo Modal", title="")
open_modal = st.button("OpenAI API Key")
if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        st.subheader('Insert OpenAI API Key')
        api_key = st.text_input(label = "Enter OpenAI API Key", type="password", placeholder="XXXXXXXXXXXXXXXXXX")
        st.session_state["OpenAPI API Key"] = api_key
        if st.button("Submit"):
            modal.close()


# Sidebar Menu
with st.sidebar:
    # Web Scrape Field and Button
    st.subheader("Enter URL")
    url = st.text_input("Enter the URL:", placeholder="https://en.wikipedia.org", label_visibility='collapsed')

    # Scraping Button:
    if st.button("Scrape URL"):
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
            href = f'<a href="data:file/csv;base64,{b64}" download="scraped_data.csv">Download CSV</a>'
            # Display the Download link for the Encoded CSV
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.write("No URL to Scrape.")

    # Save on Cloud
    st.subheader("Enter URL Title")
    url_title = st.text_input("Enter URL Title without Brackets", label_visibility='collapsed', placeholder="No Spaces")
    if st.button("Save Data"):
        # Get Data from the scraped URL
        web_dataframe = st.session_state["web_dataframe"]
        # Save the Data as Text on Cloud
        savedftxt(web_dataframe, url_title)

    st.subheader("Perform Data Embedding")
    if st.button("Data Embedding"):
        os.environ["OPENAI_API_KEY"] = st.session_state["OpenAPI API Key"]
        embedding(url_title)



# Search Query Field and Button
st.subheader("Enter Search Query")
search_query = st.text_input("Enter Search Query:", placeholder="What is Wikipedia?", label_visibility='collapsed')


# ================================ COLUMNS ===================================
col1, col2, col3, col4 = st.columns(4)
# ================================ COLUMNS ===================================

results_formating = {
    "Paragraph"     : " Show in a paragraph.",
    "Bullet Points" : " Show in bullet points. Each on a new line using <li> or <ul> tags",
    "Table"         : " Show in a table.",
    "Concise"       : " Show in a concise manner."
}

results_format = col1.selectbox('Select Format', results_formating.keys(), label_visibility='collapsed')

if col2.button("Show results"):
    os.environ["OPENAI_API_KEY"] = st.session_state["OpenAPI API Key"]
    result_formatted = search_query + results_formating.get(results_format)
    ask_results(select_url_title, result_formatted)

# Models
# gpt-3.5-turbo
# text-embedding-ada-002
# To Generate Requirements File: pip list --format=freeze > requirements.txt
# To Install Requirements from File: pip install requirements.txt