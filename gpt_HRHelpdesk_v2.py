
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import json
from pathlib import Path
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
import tempfile
import shutil

# 페이지 설정
st.set_page_config(
    page_title="HR 헬프데스크",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        flex-direction: row-reverse;
        text-align: right;
    }
    .chat-message.bot {
        background-color: #f5f5f5;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .chat-message .message {
        flex-grow: 1;
        padding: 0.5rem;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .quick-help {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# 기존 클래스들 (수정 버전)
class DocumentRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = None
        self.model_name = model_name
        self.documents = []
        self.embeddings = []
        self.doc_metadata = []
        
    def _initialize_model(self):
        """임베딩 모델 초기화 (필요시에만)"""
        if self.embedding_model is None:
            with st.spinner('임베딩 모델을 로드하고 있습니다...'):
                self.embedding_model = SentenceTransformer(self.model_name)
    
    def load_documents_from_files(self, uploaded_files):
        """업로드된 파일들에서 문서 로드"""
        self.documents = []
        self.doc_metadata = []
        
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # 파일 읽기
                content = self._read_file(Path(tmp_file_path))
                if content and len(content.strip()) > 50:
                    chunks = self._split_text(content, chunk_size=800, overlap=150)
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 30:
                            self.documents.append(chunk)
                            self.doc_metadata.append({
                                'file_name': uploaded_file.name,
                                'chunk_id': i,
                                'total_chunks': len(chunks),
                                'file_type': Path(uploaded_file.name).suffix.lower()
                            })
                    st.sidebar.success(f"✅ {uploaded_file.name}: {len(chunks)}개 청크")
                
                # 임시 파일 삭제
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.sidebar.error(f"❌ {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((idx + 1) / total_files)
        
        st.sidebar.info(f"📚 총 {len(self.documents)}개의 문서 청크 로드 완료")
        return len(self.documents) > 0
    
    def _read_file(self, file_path):
        """파일 확장자에 따라 다른 방법으로 읽기"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.txt':
                for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                return None
                
            elif extension == '.pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                    return text
                    
            elif extension == '.docx':
                doc = docx.Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
            elif extension == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
                
            elif extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, ensure_ascii=False, indent=2)
        
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            return None
        
        return None
    
    def _split_text(self, text, chunk_size=800, overlap=150):
        """텍스트를 청크로 분할"""
        chunks = []
        sections = text.split('\n\n')
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += section + '\n\n'
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = section + '\n\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_embeddings(self):
        """문서들의 임베딩 생성"""
        if not self.documents:
            return False
        
        self._initialize_model()
        with st.spinner('임베딩을 생성하고 있습니다...'):
            self.embeddings = self.embedding_model.encode(self.documents, show_progress_bar=False)
        return True
    
    def search_similar_documents(self, query, top_k=4):
        """쿼리와 유사한 문서 검색"""
        if not self.documents or len(self.embeddings) == 0:
            return []
        
        self._initialize_model()
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:
                results.append({
                    'content': self.documents[idx],
                    'similarity': similarities[idx],
                    'metadata': self.doc_metadata[idx]
                })
        
        return results

class HRHelpdeskWebApp:
    def __init__(self):
        self.rag = DocumentRAG()
        self.client = None
        self._initialize_openai()
        self.hr_term_mapping = self._setup_hr_terms()
        
    def _initialize_openai(self):
        """OpenAI 클라이언트 초기화"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            st.info("💡 .env 파일에 OPENAI_API_KEY=your_api_key_here 를 추가하세요.")
            return False
        
        try:
            self.client = OpenAI(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
            return False
    
    def _setup_hr_terms(self):
        """HR 관련 용어 매핑 설정"""
        return {
            # 휴가 관련
            "연차": ["유급휴가", "annual leave", "paid leave", "연차휴가", "년차", "휴가"],
            "병가": ["sick leave", "병휴", "병가휴가", "질병휴가"],
            "출산휴가": ["maternity leave", "육아휴직", "산전후휴가", "산휴"],
            "경조사휴가": ["bereavement leave", "경조휴가", "조의휴가"],
            "반차": ["half day", "반일휴가", "오전반차", "오후반차"],
            
            # 급여 관련
            "급여": ["salary", "월급", "임금", "봉급", "페이"],
            "상여금": ["bonus", "보너스", "상여", "성과급"],
            "야근수당": ["overtime pay", "초과근무수당", "연장근무수당", "야근비"],
            "수당": ["allowance", "지원금", "보조금"],
            "세금": ["tax", "소득세", "지방소득세", "국민연금", "건강보험"],
            
            # 복리후생 관련
            "복리후생": ["benefits", "welfare", "복지", "혜택", "지원제도"],
            "건강검진": ["health checkup", "종합검진", "의료검진"],
            "학자금": ["tuition support", "교육비", "등록금", "학비지원"],
            "동호회": ["club", "소모임", "취미활동"],
            
            # 인사평가 관련
            "인사평가": ["performance review", "평가", "고과", "성과평가"],
            "승진": ["promotion", "진급", "직급상승"],
            "교육": ["training", "연수", "교육과정", "스킬업"],
            
            # 조직 관련
            "조직도": ["organization chart", "조직구조", "부서구조"],
            "인사이동": ["personnel transfer", "전보", "발령", "부서이동"],
            "퇴직": ["resignation", "사직", "퇴사", "은퇴"],
            
            # 근무 관련
            "근무시간": ["working hours", "업무시간", "출퇴근시간"],
            "재택근무": ["remote work", "홈오피스", "원격근무", "텔레워크"],
            "유연근무": ["flexible work", "탄력근무", "선택근무"],
        }
    
    def expand_query_with_terms(self, user_input):
        """사용자 입력을 HR 용어로 확장"""
        expanded_terms = []
        original_input = user_input.lower()
        
        # 원본 입력 추가
        expanded_terms.append(user_input)
        
        # 매핑된 용어들 추가
        for key_term, related_terms in self.hr_term_mapping.items():
            if key_term in original_input:
                expanded_terms.extend(related_terms)
            else:
                # 관련 용어가 입력에 포함되어 있으면 핵심 용어도 추가
                for related_term in related_terms:
                    if related_term in original_input:
                        expanded_terms.append(key_term)
                        expanded_terms.extend(related_terms)
                        break
        
        # 중복 제거 및 확장된 쿼리 생성
        unique_terms = list(set(expanded_terms))
        
        if len(unique_terms) > 1:
            expanded_query = f"{user_input} {' '.join(unique_terms[:5])}"  # 최대 5개 용어 추가
            return expanded_query
        
        return user_input
    
    def get_system_message(self):
        """시스템 메시지 반환"""
        return {
            "role": "system", 
            "content": """당신은 전문적이고 친근한 HR(인사) 헬프데스크 담당자입니다. 
            직원들의 인사 관련 문의사항을 해결하고 정확한 정보를 제공하는 것이 주된 역할입니다.
            
            주요 담당 업무:
            • 급여, 수당, 세금 관련 문의
            • 연차, 병가, 휴가 정책 안내
            • 복리후생 제도 설명
            • 인사평가, 승진 관련 정보
            • 교육/훈련 프로그램 안내
            • 회사 정책 및 규정 설명
            • 조직 변경, 인사이동 관련 안내
            • 퇴직, 전직 관련 절차
            • 직장 내 고충 상담
            • 신입사원 온보딩 지원
            
            답변 가이드라인:
            1. 제공된 회사 HR 문서를 우선적으로 참고하여 정확한 정보 제공
            2. 민감한 개인정보는 직접 처리하지 않고 담당 부서 연결 안내
            3. 정책이나 규정은 명확하고 구체적으로 설명
            4. 절차가 복잡한 경우 단계별로 친절하게 안내
            5. 문서에 없는 내용은 일반적인 HR 지식으로 도움 제공
            6. 항상 공정하고 일관된 정보 제공
            7. 필요시 관련 부서나 담당자 연결 안내
            8. 비밀유지와 개인정보 보호 준수
            
            말투: 전문적이면서도 친근하고 이해하기 쉽게, 공감적인 톤으로 응답하되 
            회사의 품격을 유지하는 정중한 언어 사용
            
            ⭐ 중요: 다음 한국어 HR 용어들을 정확히 이해하고 응답하세요:
            - "연차" = 유급휴가, annual leave
            - "병가" = 병휴, sick leave  
            - "급여" = 월급, 임금, salary
            - "상여금" = 보너스, bonus
            - "야근수당" = 초과근무수당, overtime pay
            - "복리후생" = 복지혜택, benefits
            - "인사평가" = 성과평가, performance review
            - "승진" = 진급, promotion
            
            사용자가 "연차"라고 물어보면 유급휴가에 대한 정보를 제공하고,
            "병가"라고 물어보면 병가 정책을 안내하는 식으로 
            한국어 HR 용어를 정확히 인식하여 답변하세요."""
        }
    
    def get_context_from_documents(self, user_input):
        """사용자 입력과 관련된 HR 문서 컨텍스트 검색"""
        if not self.rag.documents:
            return ""
        
        # 사용자 입력을 HR 용어로 확장
        expanded_query = self.expand_query_with_terms(user_input)
        
        # 확장된 쿼리로 문서 검색
        similar_docs = self.rag.search_similar_documents(expanded_query, top_k=4)
        
        # 원본 쿼리로도 검색해서 결과 합치기
        if expanded_query != user_input:
            original_docs = self.rag.search_similar_documents(user_input, top_k=2)
            # 중복 제거하면서 합치기
            seen_contents = set()
            combined_docs = []
            for doc in similar_docs + original_docs:
                if doc['content'] not in seen_contents:
                    seen_contents.add(doc['content'])
                    combined_docs.append(doc)
            similar_docs = combined_docs[:4]  # 최대 4개로 제한
        
        if not similar_docs:
            return ""
        
        context = "=== 관련 HR 정책 및 문서 ===\n\n"
        for i, doc in enumerate(similar_docs):
            context += f"[참고문서 {i+1}] {doc['metadata']['file_name']}:\n"
            context += f"{doc['content']}\n"
            context += f"(관련도: {doc['similarity']:.0%})\n\n"
        
        context += "=== 위 문서를 바탕으로 답변해주세요 ===\n"
        return context, similar_docs
    
    def generate_response(self, user_input, conversation_history):
        """AI 응답 생성"""
        if not self.client:
            return "❌ OpenAI 연결에 문제가 있습니다. API 키를 확인해주세요."
        
        # 문서 컨텍스트 검색
        context_result = self.get_context_from_documents(user_input)
        context = ""
        similar_docs = []
        
        if context_result:
            context, similar_docs = context_result
        
        # 메시지 구성
        messages = [self.get_system_message()]
        
        # 이전 대화 히스토리 추가 (최근 5개만)
        for conv in conversation_history[-5:]:
            messages.append({"role": "user", "content": conv["user"]})
            messages.append({"role": "assistant", "content": conv["assistant"]})
        
        # 현재 사용자 입력
        if context:
            enhanced_input = f"{context}\n\n직원 질문: {user_input}"
        else:
            enhanced_input = user_input
        
        messages.append({"role": "user", "content": enhanced_input})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=messages,
                max_tokens=1200
            )
            
            ai_response = response.choices[0].message.content
            return ai_response, similar_docs
            
        except Exception as e:
            return f"❌ 응답 생성 중 오류가 발생했습니다: {e}", []

def main():
    # 세션 상태 초기화
    if 'app' not in st.session_state:
        st.session_state.app = HRHelpdeskWebApp()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    # 헤더
    st.title("👥 HR 헬프데스크")
    st.markdown("인사 관련 문의사항을 도와드리겠습니다! 😊")
    
    # 사이드바
    with st.sidebar:
        st.header("📁 문서 업로드")
        
        uploaded_files = st.file_uploader(
            "HR 관련 문서를 업로드하세요",
            type=['txt', 'pdf', 'docx', 'csv', 'json'],
            accept_multiple_files=True,
            help="여러 파일을 동시에 업로드할 수 있습니다."
        )
        
        if uploaded_files and not st.session_state.documents_loaded:
            if st.button("📚 문서 로드 및 임베딩 생성"):
                if st.session_state.app.rag.load_documents_from_files(uploaded_files):
                    if st.session_state.app.rag.create_embeddings():
                        st.session_state.documents_loaded = True
                        st.success("✅ 문서 로드 및 임베딩 생성 완료!")
                        st.rerun()
        
        # 문서 상태 표시
        st.markdown("---")
        st.subheader("📊 시스템 상태")
        
        if st.session_state.documents_loaded:
            st.success(f"📚 로드된 문서: {len(st.session_state.app.rag.documents)}개 청크")
        else:
            st.info("📝 문서가 로드되지 않았습니다")
        
        if st.session_state.app.client:
            st.success("🤖 OpenAI 연결: 정상")
        else:
            st.error("❌ OpenAI 연결: 실패")
        
        # 대화 히스토리 관리
        st.markdown("---")
        st.subheader("💬 대화 관리")
        
        if st.button("🗑️ 대화 초기화"):
            st.session_state.conversation_history = []
            st.rerun()
        
        st.info(f"대화 수: {len(st.session_state.conversation_history)}")
        
        # 자주 묻는 질문
        st.markdown("---")
        st.markdown("""
        <div class="quick-help">
        <h4>🔍 자주 묻는 질문</h4>
        <ul>
        <li>💰 "급여명세서는 어디서 확인하나요?"</li>
        <li>🏖️ "연차 신청은 어떻게 하나요?"</li>
        <li>🤒 "병가는 몇 일까지 사용할 수 있나요?"</li>
        <li>💰 "야근수당은 어떻게 계산되나요?"</li>
        <li>🎁 "복리후생 제도에 대해 알려주세요"</li>
        <li>📈 "인사평가 기준이 궁금해요"</li>
        <li>🏠 "재택근무 신청 방법을 알려주세요"</li>
        <li>👶 "출산휴가는 언제부터 사용하나요?"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 채팅 영역
    st.markdown("---")
    
    # 대화 히스토리 표시
    chat_container = st.container()
    
    with chat_container:
        for conv in st.session_state.conversation_history:
            # 사용자 메시지
            st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar">👤</div>
                <div class="message"><strong>직원:</strong><br>{conv['user']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # AI 응답
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="avatar">👥</div>
                <div class="message"><strong>HR 헬프데스크:</strong><br>{conv['assistant']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 참고 문서 표시
            if 'sources' in conv and conv['sources']:
                with st.expander("📋 참고 문서"):
                    for doc in conv['sources']:
                        st.write(f"📄 **{doc['metadata']['file_name']}** (관련도: {doc['similarity']:.0%})")
    
    # 사용자 입력
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "질문을 입력하세요...",
                placeholder="예: 연차 신청 방법이 궁금해요, 병가는 몇 일까지?, 야근수당 계산법",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("💬 전송", use_container_width=True)
    
    # 메시지 처리
    if submitted and user_input.strip():
        with st.spinner("답변을 생성하고 있습니다..."):
            response_data = st.session_state.app.generate_response(
                user_input, 
                st.session_state.conversation_history
            )
            
            if len(response_data) == 2:
                ai_response, similar_docs = response_data
            else:
                ai_response = response_data
                similar_docs = []
            
            # 대화 히스토리에 추가
            conversation_entry = {
                "user": user_input,
                "assistant": ai_response,
                "timestamp": datetime.now().isoformat(),
                "sources": similar_docs
            }
            
            st.session_state.conversation_history.append(conversation_entry)
            st.rerun()
    
    # 환영 메시지 (대화가 없을 때만)
    if not st.session_state.conversation_history:
        st.markdown("""
        <div class="chat-message bot">
            <div class="avatar">👥</div>
            <div class="message">
                <strong>HR 헬프데스크:</strong><br>
                안녕하세요! HR 헬프데스크입니다. 인사 관련 문의사항이 있으시면 언제든 말씀해주세요! 😊<br><br>
                💡 <strong>팁:</strong> 왼쪽 사이드바에서 HR 관련 문서를 업로드하시면 더 정확한 답변을 받을 수 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()