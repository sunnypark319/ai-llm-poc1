# PDF 문서 크기 및 처리 시간 분석 도구

import os
from pathlib import Path
import PyPDF2
import time

def analyze_pdfs(folder_path="C:/Users/SunnyPark/AI-Data/Clarios-IT"):
    """PDF 파일들의 크기와 페이지 수 분석"""
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ PDF 파일을 찾을 수 없습니다.")
        return
    
    print(f"📋 {len(pdf_files)}개의 PDF 파일을 분석합니다...")
    print("="*60)
    
    total_size = 0
    total_pages = 0
    problematic_files = []
    
    for pdf_file in pdf_files:
        try:
            # 파일 크기 확인
            file_size = pdf_file.stat().st_size
            size_mb = file_size / (1024 * 1024)
            total_size += size_mb
            
            # PDF 페이지 수 확인
            start_time = time.time()
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                
                # 첫 페이지 텍스트 추출 테스트
                try:
                    first_page_text = reader.pages[0].extract_text()
                    text_length = len(first_page_text)
                    has_text = text_length > 50
                except:
                    has_text = False
                    text_length = 0
            
            processing_time = time.time() - start_time
            total_pages += num_pages
            
            # 결과 출력
            print(f"📄 {pdf_file.name}")
            print(f"   📏 크기: {size_mb:.1f}MB")
            print(f"   📖 페이지: {num_pages}페이지")
            print(f"   ⏱️ 분석시간: {processing_time:.2f}초")
            print(f"   📝 텍스트: {'있음' if has_text else '없음/스캔본'} ({text_length}자)")
            
            # 문제가 될 수 있는 파일 체크
            if size_mb > 10:
                problematic_files.append(f"{pdf_file.name} - 크기가 큼 ({size_mb:.1f}MB)")
            if num_pages > 100:
                problematic_files.append(f"{pdf_file.name} - 페이지가 많음 ({num_pages}페이지)")
            if not has_text:
                problematic_files.append(f"{pdf_file.name} - 텍스트 추출 불가 (스캔본?)")
            if processing_time > 2:
                problematic_files.append(f"{pdf_file.name} - 처리 시간 오래 걸림 ({processing_time:.1f}초)")
            
            print()
            
        except Exception as e:
            print(f"❌ {pdf_file.name} 분석 실패: {e}")
            problematic_files.append(f"{pdf_file.name} - 파일 오류")
            print()
    
    # 요약 정보
    print("="*60)
    print("📊 요약:")
    print(f"   📁 총 파일 수: {len(pdf_files)}개")
    print(f"   📏 총 크기: {total_size:.1f}MB")
    print(f"   📖 총 페이지: {total_pages}페이지")
    
    # 예상 처리 시간
    estimated_time = total_pages * 0.1  # 페이지당 약 0.1초
    print(f"   ⏱️ 예상 처리시간: {estimated_time:.1f}초")
    
    # 문제가 있는 파일들
    if problematic_files:
        print(f"\n⚠️ 주의가 필요한 파일들:")
        for issue in problematic_files:
            print(f"   🔸 {issue}")
        
        print(f"\n💡 해결 방법:")
        print(f"   1. 큰 파일들을 작은 파일로 분할")
        print(f"   2. 스캔본 PDF는 텍스트 파일로 변환")
        print(f"   3. 불필요한 페이지 제거")
    else:
        print(f"\n✅ 모든 파일이 정상 처리 가능합니다!")

def quick_fix_slow_pdfs():
    """느린 PDF 처리를 위한 빠른 해결책"""
    print("🚀 PDF 처리 속도 개선 방법:")
    print("="*50)
    
    print("방법 1: 텍스트 파일로 변환")
    print("   - PDF → 텍스트 변환 후 .txt로 저장")
    print("   - 처리 속도 10배 빨라짐")
    print()
    
    print("방법 2: 페이지 수 제한")
    print("   - 처음 50페이지만 사용")
    print("   - 핵심 내용만 포함")
    print()
    
    print("방법 3: 임베딩 없이 실행")
    print("   - documents 폴더를 임시로 이름 변경")
    print("   - 일반 챗봇으로 즉시 실행")
    print()
    
    print("방법 4: 청크 크기 줄이기")
    print("   - chunk_size=500, overlap=100으로 설정")
    print("   - 메모리 사용량 감소")

# 실행
print("🔍 PDF 파일 분석을 시작합니다...")
analyze_pdfs()
print()
quick_fix_slow_pdfs()