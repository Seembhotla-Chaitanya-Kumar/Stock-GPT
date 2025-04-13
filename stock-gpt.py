import streamlit as st
from openai import OpenAI
import os


# Function to query and stream the response from the LLM
def stream_llm_response(model_params, api_key=None):

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o-mini",
        tools=[{
        "type": "web_search_preview",
        "user_location": {
            "type": "approximate",
            "country": "IN",
            "city": "Hyderabad",
            "region": "Telangana",
        }
    }],
        input=[
            {"role": "system", "content": "You are a helpful assistant that only talks about stock market. You can take text and aswell as PNG images to provide response"},
            *st.session_state.messages
        ],
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3
    )


    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "output_text",
                "text": response.output_text,
            }
        ]})
    
    # Yield the response text for streaming
    yield response.output_text


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
                model_max_tokens = st.number_input(label="Max Tokens",min_value= 2048, max_value=10000, value=4096)
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


        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    print(content["text"])
                    st.write(content["text"])

        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "input_text",
                        "text": prompt,
                    }]
                }
            )
            
            # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

             # Stream the assistant's response
            with st.chat_message("assistant"):
                for response_chunk in stream_llm_response(
                model_params=model_params,
                api_key=api_key
            ):  
                    st.write(response_chunk)



if __name__=="__main__":
    main()