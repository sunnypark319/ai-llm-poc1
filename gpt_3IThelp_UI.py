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
    page_title="IT 헬프데스크",
    page_icon="💻",
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
        """쿼리와 유사한 문서 검색(강화)"""
            
        if not self.documents or len(self.embeddings) == 0:
            st.write("❌ 문서나 임베딩이 없습니다.")
            return []
        
        self._initialize_model()
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 디버깅: 모든 유사도 점수 확인
        st.write(f"🔍 검색 쿼리: '{query}'")
        st.write(f"📊 유사도 점수 분포:")
        
        # 파일별 최고 유사도 확인
        file_max_scores = {}
        for i, (sim, meta) in enumerate(zip(similarities, self.doc_metadata)):
            file_name = meta['file_name']
            if file_name not in file_max_scores or sim > file_max_scores[file_name]['score']:
                file_max_scores[file_name] = {'score': sim, 'index': i}
        
        for file_name, info in file_max_scores.items():
            st.write(f"📄 {file_name}: 최고 유사도 {info['score']:.3f}")
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        #st.write(f"\n🎯 상위 {top_k}개 결과:")
        for rank, idx in enumerate(top_indices):
            score = similarities[idx]
            #st.write(f"{rank+1}. {self.doc_metadata[idx]['file_name']} - 유사도: {score:.3f}")
            
            if score > 0.1:  # 임계값을 0.2에서 0.1로 낮춤
                results.append({
                    'content': self.documents[idx],
                    'similarity': score,
                    'metadata': self.doc_metadata[idx]
                })
        
        return results
    

    def check_loaded_documents(self):  # 이 메서드를 추가
        """로드된 문서 상태 확인"""
        st.write("=== 로드된 문서 디버깅 정보 ===")
        st.write(f"총 문서 청크 수: {len(self.documents)}")
            
        # 파일별 청크 수 확인
        file_counts = {}
        for metadata in self.doc_metadata:
            file_name = metadata['file_name']
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
            
        for file_name, count in file_counts.items():
            st.write(f"📄 {file_name}: {count}개 청크")
            
        # 각 문서의 일부 내용 확인
        st.write("\n=== 문서 내용 샘플 ===")
        for i, (doc, meta) in enumerate(zip(self.documents[:5], self.doc_metadata[:5])):
            st.write(f"**청크 {i+1} ({meta['file_name']}):**")
            st.write(doc[:200] + "..." if len(doc) > 200 else doc)
            st.write("---")

class ITHelpdeskWebApp:
    def __init__(self):
        self.rag = DocumentRAG()
        self.client = None
        self._initialize_openai()
        
    def _initialize_openai(self):
        """OpenAI 클라이언트 초기화"""
        
        api_key = st.secrets["OPENAI_API_KEY"]
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
    
    def get_system_message(self):
        """시스템 메시지 반환"""
        return {
            "role": "system", 
            "content": """당신은 전문적이고 친근한 IT 헬프데스크 지원 담당자입니다. 
            직원들의 IT 관련 문제를 해결하고 기술 지원을 제공하는 것이 주된 역할입니다.
            
            주요 담당 업무:
            • 하드웨어 문제 진단 및 해결 (PC, 노트북, 프린터, 모니터 등)
            • 소프트웨어 설치, 업데이트, 오류 해결
            • 네트워크 연결 문제 (Wi-Fi, 유선, VPN)
            • 이메일 및 계정 관련 문제
            • 보안 소프트웨어 및 바이러스 관련 지원
            • 파일 공유 및 백업 문제
            • 비밀번호 재설정 및 계정 잠금 해제
            • 소프트웨어 라이선스 및 설치 가이드
            • 원격 근무 기술 지원
            • 모바일 기기 및 앱 설정
            • 화상회의 도구 사용법
            • 시스템 성능 최적화
            • 데이터 복구 및 마이그레이션
            • IT 정책 및 보안 가이드라인 안내
            
            답변 가이드라인:
            1. 제공된 IT 관련 문서를 우선적으로 참고하여 정확한 해결책 제공
            2. 단계별로 명확하고 이해하기 쉬운 해결 방법 안내
            3. 스크린샷이나 구체적인 메뉴 경로 포함하여 설명
            4. 보안 관련 사항은 반드시 강조하여 안내
            5. 문서에 없는 내용은 일반적인 IT 지식으로 도움 제공
            6. 하드웨어 교체나 복잡한 문제는 현장 지원팀 연결 안내
            7. 응급 상황 시 우선순위에 따른 대응 방법 제시
            8. 예방책과 모범 사례도 함께 안내
            9. 원격 지원이 필요한 경우 관련 도구 사용법 안내
            
            말투: 기술적이면서도 친근하고 이해하기 쉽게, 차근차근 설명하는 톤으로 응답하되 
            사용자의 기술 수준에 맞춰 적절한 용어 사용"""
        }
    
    def get_context_from_documents(self, user_input):
        """사용자 입력과 관련된 IT 문서 컨텍스트 검색"""
        if not self.rag.documents:
            return ""
        
        similar_docs = self.rag.search_similar_documents(user_input, top_k=4)
        
        if not similar_docs:
            return ""
        
        context = "=== 관련 IT 문서 및 매뉴얼 ===\n\n"
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
            enhanced_input = f"{context}\n\n사용자 문제: {user_input}"
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
        st.session_state.app = ITHelpdeskWebApp()
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    # 헤더
    st.title("💻 IT 헬프데스크")
    st.markdown("IT 관련 문제를 해결해드리겠습니다! 🛠️")
    
    # 사이드바
    with st.sidebar:
        st.header("📁 문서 업로드")
        
        uploaded_files = st.file_uploader(
            "IT 관련 문서를 업로드하세요",
            type=['txt', 'pdf', 'docx', 'csv', 'json'],
            accept_multiple_files=True,
            help="매뉴얼, 가이드, 정책 문서 등을 업로드할 수 있습니다."
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

        # 디버깅 섹션
        st.markdown("---")
        st.subheader("🔍 디버깅 도구")
        
        if st.button("📋 로드된 문서 확인"):
            if st.session_state.documents_loaded:
                st.session_state.app.rag.check_loaded_documents()
            else:
                st.warning("먼저 문서를 로드해주세요.")
        
        if st.button("🔍 검색 테스트"):
            if st.session_state.documents_loaded:
                test_query = st.text_input("테스트 검색어:", value="비밀번호 재설정")
                if test_query:
                    st.write(f"테스트 쿼리: '{test_query}'")
                    results = st.session_state.app.rag.search_similar_documents(test_query, top_k=10)
                    st.write(f"검색 결과 수: {len(results)}")
            else:
                st.warning("먼저 문서를 로드해주세요.")
        
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
        <li>🔑 "비밀번호를 재설정하고 싶어요"</li>
        <li>📶 "Wi-Fi 연결이 안 돼요"</li>
        <li>🖨️ "프린터 설정은 어떻게 하나요?"</li>
        <li>📧 "이메일 설정을 도와주세요"</li>
        <li>💾 "파일을 복구할 수 있나요?"</li>
        <li>🛡️ "바이러스에 감염된 것 같아요"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 채팅 영역
    st.markdown("---")
    
    # 대화 히스토리 표시
    for conv in st.session_state.conversation_history:
        # 사용자 메시지
        with st.chat_message("user"):
            st.write(conv['user'])
        
        # AI 응답
        with st.chat_message("assistant", avatar="💻"):
            st.write(conv['assistant'])
            
            # 참고 문서 표시
            if 'sources' in conv and conv['sources']:
                with st.expander("📋 참고 문서"):
                    for doc in conv['sources']:
                        st.write(f"📄 **{doc['metadata']['file_name']}** (관련도: {doc['similarity']:.0%})")
    
    # 사용자 입력 처리
    if prompt := st.chat_input("IT 문제를 설명해주세요... (예: 컴퓨터가 켜지지 않아요)"):
        # 사용자 메시지 즉시 표시
        with st.chat_message("user"):
            st.write(prompt)
        
        # AI 응답 생성 및 표시
        with st.chat_message("assistant", avatar="💻"):
            with st.spinner("해결책을 찾고 있습니다..."):
                response_data = st.session_state.app.generate_response(
                    prompt, 
                    st.session_state.conversation_history
                )
                
                if len(response_data) == 2:
                    ai_response, similar_docs = response_data
                else:
                    ai_response = response_data
                    similar_docs = []
                
                st.write(ai_response)
                
                # 참고 문서 표시
                if similar_docs:
                    with st.expander("📋 참고 문서"):
                        for doc in similar_docs:
                            st.write(f"📄 **{doc['metadata']['file_name']}** (관련도: {doc['similarity']:.0%})")
                
                # 대화 히스토리에 추가
                conversation_entry = {
                    "user": prompt,
                    "assistant": ai_response,
                    "timestamp": datetime.now().isoformat(),
                    "sources": similar_docs
                }
                
                st.session_state.conversation_history.append(conversation_entry)
    
    # 환영 메시지 (대화가 없을 때만)
    if not st.session_state.conversation_history:
        with st.chat_message("assistant", avatar="💻"):
            st.write("안녕하세요! IT 헬프데스크입니다. IT 관련 문제가 있으시면 언제든 말씀해주세요! 🛠️")
            st.info("💡 **팁:** 왼쪽 사이드바에서 IT 매뉴얼이나 가이드 문서를 업로드하시면 더 정확한 해결책을 받을 수 있습니다.")

if __name__ == "__main__":
    main()