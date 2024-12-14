import random
import requests
import numpy as np
import pandas as pd

import streamlit as st
from streamlit_chat import message



def request_post(url, json_body):
    response = requests.post(
                url,
                json=json_body,
            )
    return response


# Init env
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'ppls_list' not in st.session_state:
    st.session_state.ppls_list = []
if 'chat_history' not in st.session_state:
    questions = pd.read_csv('data/questions.csv')['prompt'].tolist()
    st.session_state.chat_history = []
    st.session_state.sampled_questions = random.sample(questions, 3)
if 'chosen_style_id' not in st.session_state:
    st.session_state.chosen_style_id = -1

tabs = st.tabs(["Response comparison", "Application"])
inference_url = "http://127.0.0.1:8502/gpt2"
style_pred_url = "http://127.0.0.1:8502/find_style"


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
        # HTTP POST
        try:
            response = request_post(inference_url, {"prompt": prompt, "style": style})
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


with tabs[1]:
    # Streamlit
    st.title("Application Style Chatbot Scenario")
    st.write("Please follow the chatbot!")
    
    # Init necessaries
    chat_history_expander = st.expander("Chat History", expanded=True)

    # 채팅 입력 및 버튼 설정
    with st.form("form", clear_on_submit=True):
        user_input = st.text_input("Chatbot: ", "", key="input")
        submit_button = st.form_submit_button("Send")

        with chat_history_expander:
            # User input
            if submit_button and user_input.strip() != "":
                st.session_state.chat_history.append({"content": user_input, "type": 'user'})
                if st.session_state.counter < 3:
                    response = request_post(style_pred_url, {"prompt": st.session_state.sampled_questions[st.session_state.counter], "response": user_input})
                    result = response.json()
                    st.session_state.ppls_list.append(result.get('ppls'))
                else:
                    assert not st.session_state.chosen_style_id == -1
                    response = request_post(inference_url, {"prompt": user_input, "style": style})
                    result = response.json()
                    response = result.get("response", "No response field in the server's reply.")
                    st.session_state.chat_history.append({"content": response[0], "type": "bot"})
                st.session_state.counter += 1

            # Initial three questions to find user's style
            if st.session_state.counter == 0:
                init_ment = "Hello, please answer to the 3 questions. It would be helpful to me find your style!!"
                question = st.session_state.sampled_questions[st.session_state.counter]
                if len(st.session_state.chat_history) < 2:
                    st.session_state.chat_history.append({"content": init_ment, "type": "bot"})
                    st.session_state.chat_history.append({"content": question, "type": "bot"})
            elif st.session_state.counter < 3:
                question = st.session_state.sampled_questions[st.session_state.counter]
                st.session_state.chat_history.append({"content": question, "type": "bot"})
            elif st.session_state.counter == 3:
                end_ment = "Thank you for answering all the questions! Please ask something to me!!"
                st.session_state.chat_history.append({"content": end_ment, "type": "bot"})
                st.session_state.chosen_style_id = np.argmin(np.array(st.session_state.ppls_list).mean(axis=0))
                print(np.array(st.session_state.ppls_list).mean(axis=0))
                print(f'Style ID: {st.session_state.chosen_style_id}')

            # Logging all previous conversations
            for i, past_chat in enumerate(st.session_state.chat_history):
                if past_chat['type'] == 'bot':
                    message(past_chat['content'], key=f"bot_{i}")
                else:
                    message(past_chat['content'], is_user=True, key=f"user_{i}")


    