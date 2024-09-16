
import os
import streamlit as st
from openai import OpenAI
import base64

# token = os.environ["GITHUB_TOKEN"]
# endpoint = "https://models.inference.ai.azure.com"
# model_name = "gpt-4o"

# client = OpenAI(
#     base_url=endpoint,
#     api_key=token,
# )


def page_layout():
     # configuring streamlit page settings
    # --- Page Config ---
    st.set_page_config(
        page_title="Stock-GPT",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )


    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>Stock-GPT</i> 💬</h1>""")

    # --- Side bar ---
    with st.sidebar:
        cols_keys = st.columns(1)
        with cols_keys[0]:
            default_openai_api_key = os.getenv("GITHUB_TOKEN") if os.getenv("GITHUB_TOKEN") is not None else ""  # only for development environment, otherwise it should return None
            with st.popover("🔐 OpenAI"):
                openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")

    with st.sidebar:

        model = st.selectbox("Select a model:", [
            "gpt-4o",
            "gpt-4o-mini",
        ], index=0)

        with st.popover("Model parameters"):
            model_temp = st.slider("Temperature",min_value=0.0, max_value=2.0, value=0.3, step=0.1)
            model_max_tokens = st.text_input(label="Max Tokens",value=4096, type="default")
            model_top_p = st.text_input(label="Top P", value=1, type="default")


        model_params = {

            "model": model,
            "temperature": model_temp,
            "max_tokens": model_max_tokens,
            "top_p": model_top_p
        }

    return model, openai_api_key, model_params 

def display_chat_history(openai_api_key):

    if (openai_api_key == "" or openai_api_key is None) :
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    
    else:
        if "message" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display Chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.markdown(content["text"])
                    if content["type"] == "image":
                        st.image(content["image_url"]["url"])


def stream_llm_response(model,model_params, openai_api_key,base_url="https://models.inference.ai.azure.com"): 

    client = OpenAI(base_url,openai_api_key)
    response = client.chat.completions.create(
        model=model,
        temperature= model_params["temperature"],
        max_tokens=model_params["max_tokens"],
        top_p=model_params["top_p"],
        stream=True,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only talks about stock market. You can take text and aswell as PNG images to provide response"},
            *st.session_state.chat_history
        ]
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})


    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    
    

def main():
    model, openai_api_key, model_params = page_layout()
    st.write(model)
    st.write(openai_api_key)
    st.write(model_params)
    display_chat_history(openai_api_key)
    user_prompt = st.chat_input("Ask Stock-GPT anything...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        stream_llm_response(model, openai_api_key, model_params)




if __name__=="__main__":
    main()

# # initialize chat session in streamlit if not already present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # streamlit page title
# st.title("🤖 Stock-GPT")

# # display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# # input field for user's message
# user_prompt = st.chat_input("Ask Stock-GPT...")

# # User data as a image

# if 'clicked' not in st.session_state:
#     st.session_state.clicked = False

# def set_clicked():
#     st.session_state.clicked = True

# st.button('Upload File', on_click=set_clicked)
# if st.session_state.clicked:
#     uploaded_file = st.file_uploader("Choose a PNG file", type=["png"])
#     print(uploaded_file)
#     if uploaded_file is not None:
#         base64_upload = base64.b64encode(uploaded_file.read()).decode("utf-8")


# if user_prompt:
#     # add user's message to chat and display it
#     st.chat_message("user").markdown(user_prompt)
#     st.session_state.chat_history.append({"role": "user", "content": user_prompt})

#     # send user's message to GPT-4o and get a response
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that only analyzes stock market data. Help me to understand and explain the candle stick patterns"},
#             *st.session_state.chat_history
#         ]
#     )

#     assistant_response = response.choices[0].message.content
#     st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

#     # display GPT-4o's response
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)