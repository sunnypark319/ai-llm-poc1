import os
import pdfplumber
from pathlib import Path
import time
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io

class PDFToTextConverter:
    def __init__(self):
        # 가상환경을 고려한 Tesseract 경로 설정
        self.setup_tesseract_path()
        self.setup_poppler_path()
    
    def setup_tesseract_path(self):
        """Tesseract 경로를 자동으로 찾아 설정"""
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Users\SunnyPark\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',  # 현재 설치된 경로
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
                r'C:\Tesseract-OCR\tesseract.exe'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Tesseract 경로 설정: {path}")
                    return
            
            print("⚠️ Tesseract를 찾을 수 없습니다. 수동으로 설치해주세요.")
        else:  # Mac/Linux
            # 시스템에서 tesseract 명령어 사용
            pass
    
    def setup_poppler_path(self):
        """Poppler 경로를 PATH에 추가"""
        if os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\poppler\Library\bin',
                r'C:\poppler\Library\bin',
                r'C:\tools\poppler\Library\bin'
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and path not in os.environ['PATH']:
                    os.environ['PATH'] = path + ';' + os.environ['PATH']
                    print(f"✅ Poppler 경로 설정: {path}")
                    return
    
    def extract_text_normal(self, pdf_path):
        """일반 PDF에서 텍스트 추출"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            print(f"  ⚠️ 일반 텍스트 추출 실패: {e}")
            return ""
    
    def extract_text_ocr(self, pdf_path):
        """OCR을 사용하여 이미지 PDF에서 텍스트 추출"""
        try:
            print("  🔍 OCR로 이미지 텍스트 추출 중...")
            
            # PDF를 이미지로 변환
            pages = convert_from_path(pdf_path, dpi=300)  # 높은 해상도로 변환
            
            extracted_text = ""
            total_pages = len(pages)
            
            for i, page in enumerate(pages, 1):
                print(f"    OCR 처리 중... {i}/{total_pages} 페이지")
                
                # 이미지를 OCR로 텍스트 추출
                # 한국어와 영어 동시 인식
                text = pytesseract.image_to_string(
                    page, 
                    lang='kor+eng',  # 한국어 + 영어
                    config='--oem 3 --psm 6'  # OCR 엔진 모드 설정
                )
                
                if text.strip():
                    extracted_text += f"\n=== 페이지 {i} (OCR) ===\n"
                    extracted_text += text + "\n"
            
            return extracted_text
            
        except Exception as e:
            print(f"  ❌ OCR 추출 실패: {e}")
            return ""
    
    def convert_single_pdf(self, pdf_path, output_path):
        """단일 PDF 파일을 텍스트로 변환 (일반 텍스트 + OCR)"""
        
        print(f"📄 처리 중: {Path(pdf_path).name}")
        
        # 1단계: 일반 텍스트 추출 시도
        normal_text = self.extract_text_normal(pdf_path)
        
        # 2단계: 텍스트가 부족하면 OCR 사용
        ocr_text = ""
        if len(normal_text.strip()) < 100:  # 텍스트가 너무 적으면 OCR 시도
            print("  📷 텍스트가 적어서 OCR을 시도합니다...")
            ocr_text = self.extract_text_ocr(pdf_path)
        
        # 최종 텍스트 결합
        final_text = ""
        if normal_text.strip():
            final_text += "=== 일반 텍스트 ===\n" + normal_text + "\n\n"
        
        if ocr_text.strip():
            final_text += "=== OCR 텍스트 ===\n" + ocr_text
        
        if not final_text.strip():
            print("  ❌ 추출할 수 있는 텍스트가 없습니다")
            return False
        
        # 파일로 저장
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"원본 파일: {Path(pdf_path).name}\n")
                f.write(f"변환 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(final_text)
            
            # 통계 출력
            char_count = len(final_text)
            word_count = len(final_text.split())
            print(f"  ✅ 성공! 문자 수: {char_count:,}개, 단어 수: {word_count:,}개")
            return True
            
        except Exception as e:
            print(f"  ❌ 파일 저장 실패: {e}")
            return False

def convert_folder_pdfs_to_text(input_folder, output_folder=None, use_ocr=True):
    """
    폴더 내의 모든 PDF 파일을 텍스트로 변환 (OCR 지원)
    
    Args:
        input_folder: PDF 파일들이 있는 폴더 경로
        output_folder: 텍스트 파일을 저장할 폴더 경로
        use_ocr: OCR 사용 여부
    """
    
    # 입력 폴더 확인
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {input_folder}")
        return
    
    # 출력 폴더 설정
    if output_folder is None:
        output_path = input_path / "txt_files_ocr"
    else:
        output_path = Path(output_folder)
    
    # 출력 폴더 생성
    output_path.mkdir(exist_ok=True)
    print(f"📁 출력 폴더: {output_path}")
    print(f"🔍 OCR 기능: {'활성화' if use_ocr else '비활성화'}")
    
    # PDF 파일 찾기
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ PDF 파일을 찾을 수 없습니다.")
        return
    
    print(f"📚 발견된 PDF 파일: {len(pdf_files)}개")
    print("-" * 50)
    
    # 변환기 초기화
    converter = PDFToTextConverter()
    
    # 변환 통계
    success_count = 0
    failed_files = []
    
    # 각 PDF 파일을 차례대로 변환
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}]", end=" ")
        
        output_file = output_path / f"{pdf_file.stem}.txt"
        
        if converter.convert_single_pdf(pdf_file, output_file):
            success_count += 1
        else:
            failed_files.append(pdf_file.name)
        
        print("-" * 30)
    
    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("🎉 변환 완료!")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {len(failed_files)}개")
    
    if failed_files:
        print("\n실패한 파일들:")
        for file in failed_files:
            print(f"  - {file}")
    
    print(f"\n📂 변환된 텍스트 파일 위치: {output_path}")

def install_requirements():
    """필요한 라이브러리 설치 안내"""
    print("🔧 필요한 라이브러리 설치:")
    print("-" * 30)
    print("pip install pdfplumber")
    print("pip install pytesseract")  
    print("pip install pdf2image")
    print("pip install pillow")
    print()
    print("🔧 추가 프로그램 설치:")
    print("-" * 30)
    
    if os.name == 'nt':  # Windows
        print("1. Tesseract OCR:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("   에서 Windows용 설치파일 다운로드")
        print()
        print("2. Poppler (PDF2Image용):")
        print("   https://blog.alivate.com.au/poppler-windows/")
        print("   에서 다운로드 후 PATH에 추가")
        print()
        print("3. 한국어 언어팩:")
        print("   https://github.com/tesseract-ocr/tessdata")
        print("   에서 kor.traineddata 다운로드")
    else:  # Mac/Linux
        print("1. Mac:")
        print("   brew install tesseract")
        print("   brew install poppler")
        print()
        print("2. Ubuntu/Debian:")
        print("   sudo apt install tesseract-ocr")
        print("   sudo apt install tesseract-ocr-kor")
        print("   sudo apt install poppler-utils")

# 실행 예제
if __name__ == "__main__":
    install_requirements()
    print("\n" + "=" * 50)
    
    # ========================================
    # 🔥 여기에 폴더 경로를 직접 작성하세요! 🔥
    # ========================================
    
    # 방법 1: OCR 기능 포함하여 변환
    INPUT_FOLDER = "C:/Users/SunnyPark/AI-Data/Clarios-IT"  # ← 여기에 PDF 폴더 경로 작성
    convert_folder_pdfs_to_text(INPUT_FOLDER, use_ocr=True)
    
    # 방법 2: OCR 없이 일반 텍스트만 추출
    # convert_folder_pdfs_to_text(INPUT_FOLDER, use_ocr=False)
    
    # 방법 3: 출력 폴더도 지정
    # OUTPUT_FOLDER = "C:/Users/Desktop/txt_files"
    # convert_folder_pdfs_to_text(INPUT_FOLDER, OUTPUT_FOLDER, use_ocr=True)
    
    # ========================================