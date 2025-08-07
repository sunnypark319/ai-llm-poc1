
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="HR 헬프데스크 테스트",
    page_icon="👥",
    layout="wide"
)

st.title("👥 HR 헬프데스크 업데이트 테스트")
st.write("업데이트된 코드가 정상 작동하는지 테스트 중입니다.")

# HR 용어 매핑 테스트
hr_terms = {
    "연차": ["유급휴가", "annual leave", "paid leave"],
    "병가": ["sick leave", "병휴", "질병휴가"],
    "급여": ["salary", "월급", "임금"]
}

st.subheader("🔍 HR 용어 매핑 테스트")
test_input = st.text_input("테스트할 HR 용어를 입력하세요:", placeholder="예: 연차")

if test_input:
    st.write(f"입력: {test_input}")
    
    # 용어 매핑 확인
    found_terms = []
    for key, values in hr_terms.items():
        if key in test_input.lower() or any(term in test_input.lower() for term in values):
            found_terms.extend([key] + values)
    
    if found_terms:
        st.success(f"매핑된 용어들: {', '.join(set(found_terms))}")
    else:
        st.info("매핑된 용어가 없습니다.")

st.success("✅ 기본 기능이 정상 작동합니다!")