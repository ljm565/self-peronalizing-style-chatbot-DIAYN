import os
import requests

import pandas as pd
import streamlit as st



def append_to_csv(data, file_path):
    df = pd.DataFrame([data])
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)


# Streamlit
st.title("GPT-2 Chatbot Real-time Demo")
st.write("Enter a prompt, and the server will generate a response for you!")

# User prompt and file path
st.markdown("## File path")
csv_file = st.text_input("Enter the CSV file path (e.g., data/style1.csv)", "data/style1.csv")
st.markdown("## Prompt")
prompt = st.text_input("Enter your prompt here")

# Golden label session
if "prev_golden_label" not in st.session_state:
    st.session_state.prev_golden_label = ""

# Get GPT-2's response
if prompt:
    # Endpoint URL
    server_url = "http://127.0.0.1:8502/gpt2"
    
    # HTTP POST
    try:
        response = requests.post(
            server_url,
            json={"prompt": prompt},
        )
        
        if response.status_code == 200:
            result = response.json()
            response = result.get("response", "No response field in the server's reply.")
            st.subheader("GPT-2 Response:")
            st.write(response)

            st.markdown("## Golden Label")
            golden_label = st.text_input("Enter your golden label")
            
            print('golden_label: ', golden_label)
            print('senssion state', st.session_state.prev_golden_label)
            if golden_label != st.session_state.prev_golden_label:
                tmp = {'prompt': prompt, 'response': response, 'label': golden_label}
                append_to_csv(tmp, csv_file)
                st.session_state.prev_golden_label = golden_label
            
        else:
            st.error(f"Server returned an error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the server: {e}")
