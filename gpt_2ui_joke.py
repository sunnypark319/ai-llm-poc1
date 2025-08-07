import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# 캐시 초기화 버튼
if st.button("캐시 초기화"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# 페이지 기본 설정
st.set_page_config(page_title="🃏 조커와 대화하기", layout="wide")
st.title("🃏 조커와 대화하기")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "너는 배트맨에 나오는 조커이며, 그 캐릭터에 부합하게 답변해줘"}
    ]

# 대화 출력
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages[1:]:  # system 메시지 제외
        if msg["role"] == "user":
            st.markdown(f"👤 **사용자:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"🃏 **조커:** {msg['content']}")

# 입력창 아래 고정
st.markdown("---")
user_input = st.chat_input("메시지를 입력하세요 (엔터로 전송):")

if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})

    # GPT 응답 생성
    with st.spinner("조커가 생각 중..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.9,
            messages=st.session_state.messages
        )
        ai_message = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_message})

    # 강제로 새로고침
   # st.rerun()
