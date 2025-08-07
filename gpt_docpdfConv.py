import os
import pdfplumber
from pathlib import Path
import time

def convert_folder_pdfs_to_text(input_folder, output_folder=None):
    """
    폴더 내의 모든 PDF 파일을 텍스트로 변환
    
    Args:
        input_folder: PDF 파일들이 있는 폴더 경로
        output_folder: 텍스트 파일을 저장할 폴더 경로 (기본값: input_folder/txt_files)
    """
    
    # 입력 폴더 확인
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {input_folder}")
        return
    
    # 출력 폴더 설정
    if output_folder is None:
        output_path = input_path / "txt_files"
    else:
        output_path = Path(output_folder)
    
    # 출력 폴더 생성
    output_path.mkdir(exist_ok=True)
    print(f"📁 출력 폴더: {output_path}")
    
    # PDF 파일 찾기
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ PDF 파일을 찾을 수 없습니다.")
        return
    
    print(f"📚 발견된 PDF 파일: {len(pdf_files)}개")
    print("-" * 50)
    
    # 변환 통계
    success_count = 0
    failed_files = []
    
    # 각 PDF 파일을 차례대로 변환
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] 변환 중: {pdf_file.name}")
        
        try:
            # PDF에서 텍스트 추출
            extracted_text = ""
            
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                print(f"  📄 페이지 수: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # 진행 상황 표시
                    if total_pages > 10 and page_num % 10 == 0:
                        print(f"    처리 중... {page_num}/{total_pages} 페이지")
                    
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n=== 페이지 {page_num} ===\n"
                        extracted_text += page_text + "\n"
            
            # 텍스트가 추출되었는지 확인
            if not extracted_text.strip():
                print("  ⚠️  추출된 텍스트가 없습니다 (스캔된 PDF일 가능성)")
                failed_files.append(pdf_file.name)
                continue
            
            # 텍스트 파일로 저장
            output_file = output_path / f"{pdf_file.stem}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"원본 파일: {pdf_file.name}\n")
                f.write(f"변환 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")
                f.write(extracted_text)
            
            # 통계 정보
            char_count = len(extracted_text)
            word_count = len(extracted_text.split())
            
            print(f"  ✅ 성공! → {output_file.name}")
            print(f"     문자 수: {char_count:,}개, 단어 수: {word_count:,}개")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 실패: {str(e)}")
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

def quick_convert():
    """간단한 대화형 모드"""
    print("📁 PDF → TXT 일괄 변환기")
    print("=" * 30)
    
    # 폴더 경로 입력
    while True:
        folder_path = input("PDF 파일이 있는 폴더 경로를 입력하세요: ").strip()
        
        if not folder_path:
            print("폴더 경로를 입력해주세요.")
            continue
            
        if Path(folder_path).exists():
            break
        else:
            print("❌ 폴더를 찾을 수 없습니다. 다시 입력해주세요.")
    
    # 출력 폴더 설정
    output_folder = input("출력 폴더 (엔터 입력 시 자동 설정): ").strip()
    if not output_folder:
        output_folder = None
    
    print("\n🚀 변환을 시작합니다...")
    convert_folder_pdfs_to_text(folder_path, output_folder)

# 실행 예제
if __name__ == "__main__":
    print("필요한 라이브러리 설치:")
    print("pip install pdfplumber")
    print("\n" + "=" * 50)
    
    # 직접 경로를 지정하여 실행하는 경우
    # convert_folder_pdfs_to_text("./pdf_files", "./txt_output")
    
    # 대화형 모드로 실행
    quick_convert()