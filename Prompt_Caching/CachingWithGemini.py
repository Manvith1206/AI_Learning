from google import genai
from google.genai import types
import io
import httpx
import os
import streamlit as st

client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
print("APIKEY: ", st.secrets['GEMINI_API_KEY'])
uploadede_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "csv", "txt", "docx"],
        accept_multiple_files=False
    )
import pypdf
def load_document(file_path):
        import pypdf
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

if uploadede_file:
    temp_dir = "Temp_Docs"
    os.makedirs(temp_dir, exist_ok=True)

    file_ext = os.path.splitext(uploadede_file.name)[1].lower()
    file_path = os.path.join(temp_dir, uploadede_file.name)

    with open(file_path, "wb") as f:
        f.write(uploadede_file.getbuffer())
    text = load_document(file_path)
        
chat_input = st.chat_input("Ask me anything about data communications")

if chat_input:
    model_name = "gemini-2.0-flash-001"
    system_instruction = "You are an AI assistant tasked with analyzing uploaded books and answer questions based on Book Context"

    # Create a cached content object
    cache = client.caches.create(
        model=model_name,
        config=types.CreateCachedContentConfig(
        system_instruction=system_instruction,
        contents=[text],
        )
    )

    # Display the cache details
    print(f'{cache=}')

    with st.spinner("In Progress", show_time=True):
        # Generate content using the cached prompt and document
        response = client.models.generate_content(
        model=model_name,
        contents="Please summarize this transcript",
        config=types.GenerateContentConfig(
            cached_content=cache.name
        ))

        with st.chat_message("assistant"):
            st.markdown(response.text)

    # (Optional) Print usage metadata for insights into the API call
    with st.sidebar:
        st.subheader("Response Usage MetaData")
        st.markdown(f'{response.usage_metadata=}')
