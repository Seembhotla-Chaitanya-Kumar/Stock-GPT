import streamlit as st
from openai import OpenAI
import os
from PIL import Image
import base64
from io import BytesIO
import random


# Function to query and stream the response from the LLM
def stream_llm_response(model_params, api_key=None):
    response_message = ""

    client = OpenAI(base_url="https://models.inference.ai.azure.com",api_key=api_key)
    for chunk in client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that only talks about stock market. You can take text and aswell as PNG images to provide response"},
            *st.session_state.messages
        ],
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
        max_tokens=4096,
        stream=True,
    ):
        chunk_text = chunk.choices[0].delta.content or ""
        response_message += chunk_text
        yield chunk_text


    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:

        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    
    return Image.open(BytesIO(base64.b64decode(base64_string)))



def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="Stock-GPT",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>Stock-GPT</i> 💬</h1>""")

    # --- Side Bar ---
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            default_api_key = os.getenv("GITHUB_TOKEN") if os.getenv("GITHUB_TOKEN") is not None else ""  # only for development environment, otherwise it should return None
            with st.popover("🔐 API-Key"):
                api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_api_key, type="password")

    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if (api_key == "" or api_key is None):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")

    else:

        # Side bar model options and inputs
        with st.sidebar:

            st.divider()
            
            model = st.selectbox("Select a model:", [
                "gpt-4o",
                "gpt-4o-mini",
            ], index=0)
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
                model_max_tokens = st.number_input(label="Max Tokens",min_value= 2048, max_value=8000, value=4096)
                model_top_p = st.number_input(label="Top P", value=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
                "max_tokens": model_max_tokens,
                "top_p": model_top_p
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()

            # Image Upload
            if model in ["gpt-4o", "gpt-4-turbo", "gemini-1.5-flash", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"]:
                    
                st.write(f"### **🖼️ Add an image:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            f"Upload an image:", 
                            type=["png", "jpg", "jpeg"], 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

        client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])


        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "text",
                        "text": prompt or audio_prompt,
                    }]
                }
            )
            
            # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params, 
                        api_key=api_key
                    )
                )


if __name__=="__main__":
    main()