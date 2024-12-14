import os
import requests

import pandas as pd
import streamlit as st


# Init env
tabs = st.tabs(["Response comparison", "Application"])

with tabs[0]:
    # Streamlit
    st.title("Real-time Chatbot Demo")
    st.write("Enter a prompt, and the server will generate a response for you!")


    # User prompt
    st.markdown("## Prompt")
    prompt = st.text_input("Enter your prompt here")

    # Select style
    candidates = ["Child style", "Professor style", "Philosopher style", "Nothing"]
    selected_candidate = st.radio("Choose one style:", candidates)
    st.write(f"Selected style: {selected_candidate}")
    style = candidates.index(selected_candidate) if selected_candidate != "Nothing" else -1

    # Get model response
    if prompt:
        # Endpoint URL
        server_url = "http://127.0.0.1:8502/gpt2"
        
        # HTTP POST
        try:
            response = requests.post(
                server_url,
                json={"prompt": prompt, 'style': style},
            )
            
            if response.status_code == 200:
                result = response.json()
                response = result.get("response", "No response field in the server's reply.")
                model_key = result.get("key")
                assert len(response) == len(model_key)
                for r, k in zip(response, model_key):
                    st.markdown(f"#### {k} Response:")
                    st.write(r)
                    st.write('\n'*3)

            else:
                st.error(f"Server returned an error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while connecting to the server: {e}")
