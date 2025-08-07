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
        
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                try:
                    content = self._read_file(file_path)
                    if content:
                        # 문서를 청크 단위로 분할
                        chunks = self._split_text(content, chunk_size=1000, overlap=200)
                        for i, chunk in enumerate(chunks):
                            self.documents.append(chunk)
                            self.doc_metadata.append({
                                'file_name': file_path.name,
                                'file_path': str(file_path),
                                'chunk_id': i,
                                'total_chunks': len(chunks)
                            })
                except Exception as e:
                    print(f"파일 읽기 실패 {file_path}: {e}")
        
        print(f"총 {len(self.documents)}개의 문서 청크를 로드했습니다.")
        
    def _read_file(self, file_path):
        """파일 확장자에 따라 다른 방법으로 읽기"""
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif extension == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
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
        
        return None
    
    def _split_text(self, text, chunk_size=1000, overlap=200):
        """텍스트를 청크로 분할"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # 문장 경계에서 자르기
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                cut_point = max(last_period, last_newline)
                
                if cut_point > start + chunk_size // 2:
                    chunk = text[start:cut_point + 1]
                    end = cut_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return chunks
    
    def create_embeddings(self):
        """문서들의 임베딩 생성"""
        print("임베딩 생성 중...")
        self.embeddings = self.embedding_model.encode(self.documents)
        print("임베딩 생성 완료!")
        
    def save_embeddings(self, file_path='embeddings.pkl'):
        """임베딩과 문서 데이터 저장"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.doc_metadata
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"임베딩이 {file_path}에 저장되었습니다.")
        
    def load_embeddings(self, file_path='embeddings.pkl'):
        """저장된 임베딩과 문서 데이터 로드"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.doc_metadata = data['metadata']
            print(f"임베딩이 {file_path}에서 로드되었습니다.")
            return True
        except FileNotFoundError:
            print(f"{file_path} 파일을 찾을 수 없습니다.")
            return False
    
    def search_similar_documents(self, query, top_k=3):
        """쿼리와 유사한 문서 검색"""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 상위 k개 인덱스 찾기
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'similarity': similarities[idx],
                'metadata': self.doc_metadata[idx]
            })
        
        return results

class ITSupportChatBot:
    def __init__(self, documents_folder=None):
        self.rag = DocumentRAG()
        self.messages = [
            {"role": "system", "content": """너는 전문적이고 친근한 IT 지원담당자야. 
            사용자의 IT 관련 문제를 해결하고 기술적 질문에 답변하는 것이 주된 역할이야.
            
            답변할 때 다음 사항을 지켜줘:
            1. 제공된 문서 내용을 우선적으로 참고하여 정확한 정보 제공
            2. 기술적 용어는 쉽게 설명하되 전문성 유지
            3. 단계별로 구체적인 해결 방법 제시
            4. 문서에 없는 내용은 일반적인 IT 지식으로 도움
            5. 항상 예의바르고 도움이 되는 톤으로 응답
            6. 필요시 추가 정보나 확인 사항 요청
            
            문서에서 관련 정보를 찾았다면 그것을 바탕으로 답변하고, 
            찾지 못했다면 일반적인 IT 지식으로 전문적이고 친절하게 답변해줘."""}
        ]
        
        if documents_folder:
            self._setup_documents(documents_folder)
    
    def _setup_documents(self, folder_path):
        """문서 설정 및 임베딩 생성/로드"""
        # 기존 임베딩이 있는지 확인
        if not self.rag.load_embeddings():
            print("새로운 임베딩을 생성합니다...")
            self.rag.load_documents(folder_path)
            if self.rag.documents:
                self.rag.create_embeddings()
                self.rag.save_embeddings()
            else:
                print("로드된 문서가 없습니다. 일반 챗봇으로 실행됩니다.")
    
    def get_context_from_documents(self, user_input, top_k=3):
        """사용자 입력과 관련된 문서 컨텍스트 검색"""
        if not self.rag.documents:
            return ""
        
        similar_docs = self.rag.search_similar_documents(user_input, top_k=top_k)
        
        context = "관련 문서 내용:\n"
        for i, doc in enumerate(similar_docs):
            if doc['similarity'] > 0.3:  # 유사도 임계값
                context += f"\n[문서 {i+1}] {doc['metadata']['file_name']}:\n"
                context += f"{doc['content'][:500]}...\n"
        
        return context if len(context) > 20 else ""
    
    def chat(self):
        """IT 지원 챗봇 실행"""
        print("🖥️  IT 지원 챗봇입니다! 기술적 문제나 질문이 있으시면 언제든 말씀해주세요.")
        print("(종료하려면 'exit' 입력)")
        print("=" * 60)
        
        while True:
            user_input = input("\n👤 사용자: ")
            if user_input.lower() == "exit":
                print("🖥️ IT 지원: 문제 해결에 도움이 되었기를 바랍니다. 언제든 다시 문의해주세요! 😊")
                break
            
            # 문서에서 관련 컨텍스트 검색
            context = self.get_context_from_documents(user_input)
            
            # 컨텍스트가 있으면 메시지에 추가
            if context:
                enhanced_input = f"{context}\n\n사용자 질문: {user_input}"
            else:
                enhanced_input = user_input
            
            # 대화 내역에 사용자 메시지 추가
            self.messages.append({"role": "user", "content": enhanced_input})
            
            try:
                # OpenAI API 호출
                response = client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.7,  # IT 지원은 조금 더 일관된 답변을 위해 낮춤
                    messages=self.messages,
                    max_tokens=1000
                )
                
                # AI 응답 저장 및 출력
                ai_message = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": ai_message})
                print(f"\n🖥️ IT 지원: {ai_message}")
                
                # 컨텍스트가 사용되었다면 출처 표시
                if context:
                    print("\n📋 [참고 문서]")
                    similar_docs = self.rag.search_similar_documents(user_input, top_k=3)
                    for doc in similar_docs:
                        if doc['similarity'] > 0.3:
                            print(f"   📄 {doc['metadata']['file_name']} (관련도: {doc['similarity']:.0%})")
                
            except Exception as e:
                print(f"❌ 시스템 오류가 발생했습니다: {e}")
                print("🔧 잠시 후 다시 시도해주세요.")

# 사용 예시
if __name__ == "__main__":
    # 🔥 여기서 문서 폴더 경로를 수정하세요! 🔥
    # 예시들:
    # documents_folder = "./documents"           # 현재 폴더 안의 documents 폴더
    documents_folder = "C:/Users/SunnyPark/AI-Data/Clarios-IT/txt_files_ocr"  # 윈도우 절대경로
    # documents_folder = "/Users/사용자명/Documents/내문서들"    # 맥 절대경로
    # documents_folder = None                    # 문서 없이 일반 챗봇으로 실행
    
   # documents_folder = "./documents"  # 👈 이 부분을 수정하세요!
    
    # 폴더가 없으면 생성하라고 안내
    if documents_folder and not os.path.exists(documents_folder):
        print(f"문서 폴더 '{documents_folder}'가 없습니다.")
        print("다음 중 하나를 선택하세요:")
        print("1. 해당 폴더를 만들고 문서들(.txt, .pdf, .docx, .csv, .json)을 넣어주세요.")
        print("2. 또는 코드에서 documents_folder 변수를 올바른 경로로 수정하세요.")
        print("3. 또는 문서 없이 실행하려면 documents_folder = None으로 설정하세요.")
        
        create_folder = input("폴더를 자동으로 생성하시겠습니까? (y/n): ").lower()
        if create_folder == 'y':
            os.makedirs(documents_folder)
            print(f"'{documents_folder}' 폴더가 생성되었습니다. 문서들을 넣고 다시 실행하세요.")
            exit()
        else:
            documents_folder = None
    
    # 챗봇 실행
    chatbot = ITSupportChatBot(documents_folder)
    chatbot.chat()