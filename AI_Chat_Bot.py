from google import genai
import streamlit as st
from google.genai import types
from PIL import Image
from io import BytesIO
import Constants as Constants
from PyPDF2 import PdfReader
st.title(Constants.AI_Application_Title)

st.markdown("""
<style>
.stChatInputContainer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: white;
    padding: 1rem;
    z-index: 100;
    border-top: 1px solid #e6e6e6;
}
.main .block-container {
    padding-bottom: 80px; /* Add padding to prevent content from being hidden behind the chat input */
}
</style>
""", unsafe_allow_html=True)


client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
context_file = None

Chat_Interface, Edit_Image, Generate_Images = st.tabs([
    Constants.Chat_Interface, 
    Constants.Edit_Image_Text,
    Constants.Generate_Image_Text,
    "Context"])

if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = Constants.AI_Model_Name_For_Chat

# with ContextCreater:
#     context_file = st.file_uploader(label="Upload a image",
#                      type=["pdf"])

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text[:4000]  # Ensure within token limit

with Chat_Interface:
    text_prompt = st.chat_input(Constants.Chat_Input_Message)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if context_file:
        context_file = extract_text_from_pdf(context_file)

    if text_prompt:
        st.session_state.messages.append({"role": "user", "content": text_prompt})

        with st.chat_message("user"):
            st.markdown(text_prompt)

        # Generate Gemini response
        with st.chat_message("assistant"):
            with st.spinner("Generating"):
                if context_file:
                    response = client.models.generate_content(
                        model=Constants.AI_Model_Name_For_Chat,
                        contents= f"Answer based on the following document:\n\n{context_file}\n\nUser's question: {text_prompt}")
                else:
                    response = client.models.generate_content(
                        model=Constants.AI_Model_Name_For_Chat,
                        contents=[f'{Constants.Chat_Extra_Context} {text_prompt}'])
                st.markdown(response.text)
    
        # Store assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": response.text})
with Edit_Image:
    # resp_contents = ('Hi, can you create 3d rendered image of soldier of powerful nation and this soldier is powerful of all with gold and black armor with powerful weapons dont keep too much weapons add weapons so that he can hold')
    img_edit_prompt = st.chat_input(Constants.Image_edit_Message)
    file = st.file_uploader(label="Upload a image",
                     type=["jpg", "jpeg", "png"])
    
    if img_edit_prompt and file:
        image = Image.open(file)

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[img_edit_prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save("gemini-native-image.png")
                image.show()
with Generate_Images:
    if "images" not in st.session_state:
        st.session_state.images = []
    if "latest_image" not in st.session_state:
        st.session_state.latest_image = None
    
    for message in st.session_state.images:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="Generated Image")
                
    # resp_contents = ('Hi, can you create 3d rendered image of soldier of powerful nation and this soldier is powerful of all with gold and black armor with powerful weapons dont keep too much weapons add weapons so that he can hold')
    img_prompt = st.text_input(Constants.Image_Generation_Message)

    with st.spinner("Generating"):
        if img_prompt:
            st.session_state.images.append({"role": "user", "content": img_prompt, "type": "text"})

            with st.chat_message("user"):
                st.markdown(img_prompt)
            
            if st.session_state.latest_image:    
                response = client.models.generate_content(
                    model=Constants.AI_Model_Name_For_Image_Generation,
                    contents=[f'{Constants.Image_Generation_Extra_Context} {img_prompt}', st.session_state.latest_image],
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image']
                    )
                )
            else:
                response = client.models.generate_content(
                    model = Constants.AI_Model_Name_For_Image_Generation,
                    contents=f'{Constants.Image_Generation_Extra_Context} {img_prompt}',
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"]
                    )

                )

            if response:
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        st.image(image)
                        st.session_state.images.append({"role": "assistant", "content": image, "type": "image"})
                        st.session_state.latest_image = image





