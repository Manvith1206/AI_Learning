import anthropic
import streamlit as st
import os

client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])

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
    with st.spinner("In Progress", show_time=True):

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing uploaded books and answer questions based on Book Context."
                },
                {
                "type": "text",
                "text": f"Here is the full text of a Data Communications Chapter: {text}",
                "cache_control": {"type": "ephemeral"}
            }
            ],
            
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "user",
                    "content": chat_input
                }
            ]
        )

        with st.chat_message("assistant"):
            st.markdown(response.content[0].text)
        print(response.model_dump_json())
