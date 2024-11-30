import streamlit as st
import requests

# Streamlit 앱 구성
st.title("GPT-2 Chatbot with Streamlit")
st.write("Enter a prompt, and the server will generate a response for you!")

# 사용자 입력 프롬프트
prompt = st.text_area("Enter your prompt here:")

# 응답 생성 버튼
if st.button("Generate Response"):
    if prompt.strip():
        # 서버 엔드포인트 URL
        server_url = "http://127.0.0.1:8002/gpt2"
        
        # HTTP POST 요청 보내기
        try:
            response = requests.post(
                server_url,
                json={"prompt": prompt},  # JSON 형태로 프롬프트 전달
                timeout=10  # 타임아웃 설정 (초)
            )
            
            if response.status_code == 200:
                # 서버로부터의 응답 처리
                result = response.json()
                generated_text = result.get("response", "No response field in the server's reply.")
                st.subheader("GPT-2 Response:")
                st.write(generated_text)
            else:
                st.error(f"Server returned an error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while connecting to the server: {e}")
    else:
        st.warning("Please enter a prompt to generate a response!")
