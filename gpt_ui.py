import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 세션 상태에 메시지 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "너는 배트맨에 나오는 조커이며, 그 캐릭터에 부합하게 답변해줘"}
    ]
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# 입력창 UI (placeholder는 선택사항)
user_input = st.text_input("입력:", key="user_input")

# 입력이 들어오면 처리
if user_input:
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_log.append(f"👤 사용자: {user_input}")

    # GPT 응답 생성
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=st.session_state.messages
    )
    ai_response = response.choices[0].message.content.strip()

    # 응답 저장 및 출력
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.session_state.chat_log.append(f"🤖 AI: {ai_response}")

    # 입력창 초기화
    st.session_state.user_input = ""

# 채팅 로그 표시 (최근 대화 먼저 보고 싶으면 reversed 사용)
for chat in st.session_state.chat_log:
    st.markdown(chat)