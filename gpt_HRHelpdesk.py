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

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class DocumentRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.doc_metadata = []
        
    def load_documents(self, folder_path):
        """폴더에서 다양한 형태의 문서들을 읽어옴"""
        folder = Path(folder_path)
        print(f"📁 문서 폴더 스캔 중: {folder_path}")
        
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                try:
                    content = self._read_file(file_path)
                    if content and len(content.strip()) > 50:  # 최소 길이 체크
                        # 문서를 청크 단위로 분할
                        chunks = self._split_text(content, chunk_size=800, overlap=150)
                        for i, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 30:  # 너무 짧은 청크 제외
                                self.documents.append(chunk)
                                self.doc_metadata.append({
                                    'file_name': file_path.name,
                                    'file_path': str(file_path),
                                    'chunk_id': i,
                                    'total_chunks': len(chunks),
                                    'file_type': file_path.suffix.lower()
                                })
                        print(f"✅ {file_path.name}: {len(chunks)}개 청크 로드")
                except Exception as e:
                    print(f"❌ 파일 읽기 실패 {file_path.name}: {e}")
        
        print(f"📚 총 {len(self.documents)}개의 문서 청크를 로드했습니다.")
        
        # 첫 번째 문서 미리보기
        if self.documents:
            print(f"📋 첫 번째 문서 미리보기: {self.documents[0][:150]}...")
        
    def _read_file(self, file_path):
        """파일 확장자에 따라 다른 방법으로 읽기"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.txt':
                # 여러 인코딩 시도
                for encoding in ['utf-8', 'cp949', 'euc-kr', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            return content
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
            print(f"파일 읽기 오류 {file_path.name}: {e}")
            return None
        
        return None
    
    def _split_text(self, text, chunk_size=800, overlap=150):
        """텍스트를 청크로 분할 (HR 문서에 최적화)"""
        chunks = []
        
        # 섹션별로 먼저 분할 (HR 문서는 보통 섹션이 있음)
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
        
        # 너무 긴 청크는 추가 분할
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                # 긴 청크를 문장 단위로 분할
                sentences = chunk.split('. ')
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) < chunk_size:
                        temp_chunk += sentence + '. '
                    else:
                        if temp_chunk.strip():
                            final_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + '. '
                if temp_chunk.strip():
                    final_chunks.append(temp_chunk.strip())
        
        return final_chunks
    
    def create_embeddings(self):
        """문서들의 임베딩 생성"""
        if not self.documents:
            print("❌ 임베딩할 문서가 없습니다.")
            return
            
        print("🧠 임베딩 생성 중...")
        self.embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        print("✅ 임베딩 생성 완료!")
        
    def save_embeddings(self, file_path='hr_embeddings.pkl'):
        """임베딩과 문서 데이터 저장"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.doc_metadata,
            'created_time': datetime.now().isoformat()
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 임베딩이 {file_path}에 저장되었습니다.")
        
    def load_embeddings(self, file_path='hr_embeddings.pkl'):
        """저장된 임베딩과 문서 데이터 로드"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.doc_metadata = data['metadata']
            created_time = data.get('created_time', '알 수 없음')
            print(f"📂 임베딩 로드 완료: {file_path}")
            print(f"📅 생성 시간: {created_time}")
            print(f"📚 문서 수: {len(self.documents)}개")
            return True
        except FileNotFoundError:
            print(f"📁 {file_path} 파일을 찾을 수 없습니다. 새로 생성합니다.")
            return False
        except Exception as e:
            print(f"❌ 임베딩 로드 실패: {e}")
            return False
    
    def search_similar_documents(self, query, top_k=4):
        """쿼리와 유사한 문서 검색"""
        if not self.documents or len(self.embeddings) == 0:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 상위 k개 인덱스 찾기
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # HR 관련성 임계값
                results.append({
                    'content': self.documents[idx],
                    'similarity': similarities[idx],
                    'metadata': self.doc_metadata[idx]
                })
        
        return results

class HRHelpdeskChatBot:
    def __init__(self, documents_folder=None):
        self.rag = DocumentRAG()
        self.conversation_history = []
        self.messages = [
            {"role": "system", "content": """당신은 전문적이고 친근한 HR(인사) 헬프데스크 담당자입니다. 
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
            
            문서에서 관련 정보를 찾았다면 그것을 바탕으로 답변하고, 
            찾지 못했다면 일반적인 HR 지식과 모범 사례로 도움을 드리되 
            정확한 회사 정책은 담당 부서에 확인하도록 안내해주세요."""}
        ]
        
        if documents_folder:
            self._setup_documents(documents_folder)
    
    def _setup_documents(self, folder_path):
        """문서 설정 및 임베딩 생성/로드"""
        print(f"📁 HR 문서 폴더: {folder_path}")
        print(f"📂 폴더 존재 여부: {os.path.exists(folder_path)}")
        
        # 기존 임베딩이 있는지 확인
        if not self.rag.load_embeddings():
            print("🔄 새로운 임베딩을 생성합니다...")
            self.rag.load_documents(folder_path)
            if self.rag.documents:
                self.rag.create_embeddings()
                self.rag.save_embeddings()
            else:
                print("⚠️ 로드된 문서가 없습니다. 일반 HR 헬프데스크로 실행됩니다.")
    
    def get_context_from_documents(self, user_input, top_k=4):
        """사용자 입력과 관련된 HR 문서 컨텍스트 검색"""
        if not self.rag.documents:
            return ""
        
        similar_docs = self.rag.search_similar_documents(user_input, top_k=top_k)
        
        if not similar_docs:
            return ""
        
        context = "=== 관련 HR 정책 및 문서 ===\n\n"
        for i, doc in enumerate(similar_docs):
            context += f"[참고문서 {i+1}] {doc['metadata']['file_name']}:\n"
            context += f"{doc['content']}\n"
            context += f"(관련도: {doc['similarity']:.0%})\n\n"
        
        context += "=== 위 문서를 바탕으로 답변해주세요 ===\n"
        return context
    
    def save_conversation(self, user_input, ai_response):
        """대화 내역 저장"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response
        }
        self.conversation_history.append(conversation)
        
        # 파일로 저장 (선택사항)
        try:
            with open('hr_conversation_log.json', 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except:
            pass  # 저장 실패해도 계속 진행
    
    def get_quick_help(self):
        """자주 묻는 질문 가이드"""
        return """
🔍 HR 헬프데스크 - 자주 묻는 질문

💰 급여/수당 관련:
   • "급여명세서는 어디서 확인하나요?"
   • "야근수당 신청 방법이 궁금해요"
   • "세금공제 항목에 대해 알려주세요"

🏖️ 휴가/연차 관련:
   • "연차 신청은 어떻게 하나요?"
   • "병가 사용 조건이 궁금해요"
   • "출산휴가 기간은 얼마나 되나요?"

🎁 복리후생:
   • "건강검진 일정을 알려주세요"
   • "자녀 학자금 지원 제도가 있나요?"
   • "직원 할인 혜택에 대해 궁금해요"

📈 인사평가/승진:
   • "인사평가 기준이 궁금해요"
   • "승진 요건은 무엇인가요?"
   • "교육 프로그램 신청 방법을 알려주세요"

원하시는 주제로 언제든 질문해주세요! 🙂
        """
    
    def chat(self):
        """HR 헬프데스크 챗봇 실행"""
        print("👥 HR 헬프데스크 챗봇입니다!")
        print("인사 관련 문의사항이 있으시면 언제든 말씀해주세요. 😊")
        print("=" * 60)
        
        # 도움말 표시
        print(self.get_quick_help())
        print("💬 대화를 시작하려면 질문을 입력하세요.")
        print("종료하려면 'exit' 또는 '종료'를 입력하세요.")
        print("도움말을 다시 보려면 'help' 또는 '도움말'을 입력하세요.")
        print("=" * 60)
        
        while True:
            user_input = input("\n👤 직원: ").strip()
            
            if user_input.lower() in ["exit", "종료", "quit"]:
                print("\n👥 HR 헬프데스크: 문의해주셔서 감사합니다! 추가 문의사항이 있으시면 언제든 연락주세요. 😊")
                break
            
            if user_input.lower() in ["help", "도움말", "도움"]:
                print(self.get_quick_help())
                continue
            
            if not user_input:
                print("질문을 입력해주세요.")
                continue
            
            # 문서에서 관련 컨텍스트 검색
            context = self.get_context_from_documents(user_input)
            
            # 컨텍스트가 있으면 메시지에 추가
            if context:
                enhanced_input = f"{context}\n\n직원 질문: {user_input}"
            else:
                enhanced_input = user_input
            
            # 대화 내역에 사용자 메시지 추가
            self.messages.append({"role": "user", "content": enhanced_input})
            
            try:
                # OpenAI API 호출
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.3,  # HR은 일관되고 정확한 답변이 중요
                    messages=self.messages,
                    max_tokens=1200
                )
                
                # AI 응답 저장 및 출력
                ai_message = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": ai_message})
                print(f"\n👥 HR 헬프데스크: {ai_message}")
                
                # 컨텍스트가 사용되었다면 출처 표시
                if context:
                    print("\n📋 [참고 문서]")
                    similar_docs = self.rag.search_similar_documents(user_input, top_k=3)
                    for doc in similar_docs:
                        if doc['similarity'] > 0.2:
                            print(f"   📄 {doc['metadata']['file_name']} (관련도: {doc['similarity']:.0%})")
                
                # 대화 저장
                self.save_conversation(user_input, ai_message)
                
                # 메시지 히스토리 관리 (너무 길어지지 않도록)
                if len(self.messages) > 20:
                    # 시스템 메시지와 최근 대화만 유지
                    self.messages = [self.messages[0]] + self.messages[-15:]
                
            except Exception as e:
                print(f"❌ 시스템 오류가 발생했습니다: {e}")
                print("🔧 잠시 후 다시 시도해주시거나, 담당 부서로 직접 연락해주세요.")

# 사용 예시
if __name__ == "__main__":
    # 🔥 여기서 HR 문서 폴더 경로를 수정하세요! 🔥
    hr_documents_folder = "C:/Users/SunnyPark/AI-Data/Clarios-HR"  # HR 문서 폴더 경로
    
    # 다른 경로 예시들:
    # hr_documents_folder = "./hr_documents"           # 현재 폴더 안의 hr_documents 폴더
    # hr_documents_folder = "D:/Company/HR_Policies"   # 다른 드라이브의 HR 정책 폴더
    # hr_documents_folder = None                       # 문서 없이 일반 HR 헬프데스크로 실행
    
    # 폴더가 없으면 생성하라고 안내
    if hr_documents_folder and not os.path.exists(hr_documents_folder):
        print(f"📁 HR 문서 폴더 '{hr_documents_folder}'가 없습니다.")
        print("\n다음 중 하나를 선택하세요:")
        print("1. 해당 폴더를 만들고 HR 문서들(.txt, .pdf, .docx, .csv, .json)을 넣어주세요.")
        print("2. 또는 코드에서 hr_documents_folder 변수를 올바른 경로로 수정하세요.")
        print("3. 또는 문서 없이 실행하려면 hr_documents_folder = None으로 설정하세요.")
        
        create_folder = input("\n폴더를 자동으로 생성하시겠습니까? (y/n): ").lower()
        if create_folder == 'y':
            os.makedirs(hr_documents_folder)
            print(f"✅ '{hr_documents_folder}' 폴더가 생성되었습니다.")
            print("HR 관련 문서들을 넣고 다시 실행하세요.")
            exit()
        else:
            hr_documents_folder = None
    
    print("🏢 HR 헬프데스크 챗봇을 시작합니다...")
    print("=" * 60)
    
    # HR 헬프데스크 챗봇 실행
    chatbot = HRHelpdeskChatBot(hr_documents_folder)
    chatbot.chat()