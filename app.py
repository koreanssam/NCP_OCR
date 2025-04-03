import streamlit as st
import os
import tempfile
import uuid
import time
import json
import requests
import base64
from pathlib import Path
from io import BytesIO
import shutil

# --- 세션 상태 초기화 ---
if 'extracted_text' not in st.session_state: st.session_state.extracted_text = ""
if 'processing_done' not in st.session_state: st.session_state.processing_done = False
if 'last_processed_type' not in st.session_state: st.session_state.last_processed_type = None

# --- 네이버 OCR 설정 ---
NAVER_OCR_API_URL = st.secrets.get("NAVER_OCR_API_URL")
NAVER_OCR_SECRET_KEY = st.secrets.get("NAVER_OCR_SECRET_KEY")

# --- PDF 처리 클래스 ---
class ClovaOCRProcessor:
    def __init__(self, api_url, secret_key):
        self.secret_key = secret_key
        self.api_url = api_url
        self.chunk_size = 10  # 10개씩 처리할 때마다 딜레이 적용

    def process_pdf(self, pdf_path):
        """단일 PDF 파일을 OCR 처리하여 텍스트 추출"""
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        # PDF 파일을 Base64로 인코딩
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

        # OCR API 요청 준비
        image_info = [{
            'format': 'pdf',
            'name': Path(pdf_path).name,
            'data': base64_pdf
        }]

        # OCR API 호출
        ocr_result = self._call_ocr_api(image_info)

        # OCR 결과에서 텍스트 추출
        if ocr_result and 'images' in ocr_result and len(ocr_result['images']) > 0:
            all_page_texts = []
            for idx, image_result in enumerate(ocr_result['images']):
                page_text = self._extract_text(image_result)
                all_page_texts.append(f"## 페이지 {idx+1}\n\n{page_text}")
            return "\n\n".join(all_page_texts)
        else:
            return "OCR 결과를 불러오지 못했습니다."

    def _call_ocr_api(self, image_info):
        """네이버 클로바 OCR API를 호출"""
        request_json = {
            'version': 'V2',
            'requestId': str(uuid.uuid4()),
            'timestamp': int(round(time.time() * 1000)),
            'images': image_info
        }
        headers = {
            'Content-Type': 'application/json',
            'X-OCR-SECRET': self.secret_key
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=request_json,
                timeout=100
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API 호출 오류: {str(e)}")
            return None

    def _extract_text(self, image_result):
        """OCR 결과에서 텍스트를 추출하고 마크다운 형식으로 구조화"""
        if 'fields' not in image_result:
            return ""
        
        # 필드 정보를 활용하여 구조화된 텍스트 생성
        fields = image_result['fields']
        
        # 텍스트 위치 정보를 활용하여 줄바꿈 및 구조화 처리
        lines = []
        current_line = []
        last_y = None
        last_x_end = None
        y_threshold = 10  # 같은 줄로 인식할 y좌표 차이 임계값
        x_gap_threshold = 50  # 같은 줄에서 단어 간격으로 인식할 x좌표 차이 임계값
        line_height = None  # 줄 높이 (문단 구분에 사용)
        
        # # 머릿말과 꼬릿말로 간주할 영역의 임계값
        # header_y_max = 150    # 페이지 상단 150 픽셀 이내는 머릿말로 간주
        # footer_y_min = 2300   # 페이지 하단 2300 픽셀 이후는 꼬릿말로 간주
        
        for field in fields:
            text = field.get('inferText', '')
            vertices = field.get('boundingPoly', {}).get('vertices', [])
            
            if vertices and len(vertices) > 0:
                # 좌표 정보 추출
                x_coords = [vertex.get('x', 0) for vertex in vertices]
                y_coords = [vertex.get('y', 0) for vertex in vertices]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                current_y = min(y_coords)
                current_x_start = min(x_coords)
                current_x_end = max(x_coords)
                
                # 첫 필드에서 줄 높이 계산
                if line_height is None:
                    line_height = height
                
                # 수평 텍스트만 추출 (폭이 높이보다 클 때)
                if width < height:
                    continue
                
                # # 머릿말/꼬릿말 제거
                # if max(y_coords) < header_y_max or min(y_coords) > footer_y_min:
                #     continue
                
                # 새로운 줄 또는 문단 시작 여부 확인
                new_line = False
                new_paragraph = False
                
                if last_y is not None:
                    y_diff = abs(current_y - last_y)
                    
                    # 같은 줄인지 확인 (y좌표 차이가 임계값 이내)
                    if y_diff > y_threshold:
                        new_line = True
                        
                        # 문단 구분 확인 (줄 높이의 1.5배 이상 차이나면 새 문단으로 간주)
                        if y_diff > line_height * 1.5:
                            new_paragraph = True
                    
                    # 같은 줄이지만 x좌표 간격이 너무 큰 경우 (들여쓰기 등) 새 줄로 간주
                    elif last_x_end is not None and (current_x_start - last_x_end) > x_gap_threshold:
                        new_line = True
                
                if new_line:
                    # 현재 줄 저장
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = []
                    
                    # 문단 구분을 위한 빈 줄 추가
                    if new_paragraph:
                        lines.append('')
                
                current_line.append(text)
                last_y = current_y
                last_x_end = current_x_end
            else:
                current_line.append(text)
        
        # 마지막 줄 처리
        if current_line:
            lines.append(' '.join(current_line))
        
        # 문장 및 문단 연결 처리
        processed_lines = []
        current_paragraph = []
        
        for i, line in enumerate(lines):
            # 빈 줄은 문단 구분자로 처리
            if not line.strip():
                if current_paragraph:
                    # 문단 내 문장들을 연결하여 하나의 문단으로 만듦
                    processed_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                processed_lines.append('')
                continue
            
            # 문장 끝 판단 (마침표, 물음표, 느낌표로 끝나는 경우)
            ends_with_sentence = line.rstrip().endswith(('.', '?', '!', '"', "'", '」', '』', '》', '）', '】', ')'))
            
            # 다음 줄이 대문자나 들여쓰기로 시작하는지 확인 (새 문장 시작 여부)
            next_line_new_sentence = False
            if i < len(lines) - 1 and lines[i+1].strip():
                next_line = lines[i+1].strip()
                if next_line[0].isupper() or next_line.startswith('  '):
                    next_line_new_sentence = True
            
            # 현재 줄을 문단에 추가
            current_paragraph.append(line)
            
            # 문장이 끝나고 다음 줄이 새 문장이면 문단 완성
            if ends_with_sentence and (next_line_new_sentence or i == len(lines) - 1):
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # 남은 문단 처리
        if current_paragraph:
            processed_lines.append(' '.join(current_paragraph))
        
        # 마크다운 형식으로 구조화
        markdown_text = ""
        in_table = False
        
        for i, line in enumerate(processed_lines):
            if not line.strip():
                markdown_text += "\n"
                in_table = False
                continue
            
            # 제목으로 보이는 텍스트는 마크다운 제목 형식으로 변환
            if len(line) < 50 and (i == 0 or not processed_lines[i-1].strip()):
                if i < len(processed_lines) - 1 and processed_lines[i+1].strip():
                    markdown_text += f"### {line}\n\n"
                    continue
            
            # 표로 보이는 텍스트는 마크다운 표 형식으로 변환
            if "|" in line or "\t" in line:
                cells = line.split("|") if "|" in line else line.split("\t")
                if not in_table:
                    in_table = True
                    markdown_text += "| " + " | ".join(cells) + " |\n"
                    markdown_text += "| " + " | ".join(["---" for _ in cells]) + " |\n"
                else:
                    markdown_text += "| " + " | ".join(cells) + " |\n"
            else:
                in_table = False
                markdown_text += line + "\n\n"
        
        return markdown_text

# --- 이미지 OCR 함수 ---
def call_naver_ocr_image(image_bytes, image_format):
    """이미지 파일에 대한 네이버 OCR API 호출"""
    if not NAVER_OCR_API_URL or not NAVER_OCR_SECRET_KEY:
        return None, "OCR API URL 또는 Secret Key가 설정되지 않았습니다."

    request_json = {
        'images': [{'format': image_format, 'name': 'ocr_image'}],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [('file', image_bytes)]
    headers = {'X-OCR-SECRET': NAVER_OCR_SECRET_KEY}

    try:
        response = requests.request("POST", NAVER_OCR_API_URL, headers=headers, data=payload, files=files, timeout=30)
        response.raise_for_status()

        result_json = response.json()
        
        # 마크다운 형식으로 변환
        markdown_text = ""
        if result_json.get('images') and len(result_json['images']) > 0:
            if 'fields' in result_json['images'][0]:
                # 필드 정보를 활용하여 구조화된 텍스트 생성
                fields = result_json['images'][0]['fields']
                
                # 텍스트 위치 정보를 활용하여 줄바꿈 및 문단 처리
                lines = []
                current_line = []
                last_y = None
                last_x_end = None
                y_threshold = 10  # 같은 줄로 인식할 y좌표 차이 임계값
                x_gap_threshold = 50  # 같은 줄에서 단어 간격으로 인식할 x좌표 차이 임계값
                line_height = None  # 줄 높이 (문단 구분에 사용)
                
                for field in fields:
                    text = field.get('inferText', '')
                    vertices = field.get('boundingPoly', {}).get('vertices', [])
                    
                    if vertices and len(vertices) > 0:
                        x_coords = [vertex.get('x', 0) for vertex in vertices]
                        y_coords = [vertex.get('y', 0) for vertex in vertices]
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        current_y = min(y_coords)
                        current_x_start = min(x_coords)
                        current_x_end = max(x_coords)
                        
                        # 첫 필드에서 줄 높이 계산
                        if line_height is None:
                            line_height = height
                        
                        # 수평 텍스트만 추출 (폭이 높이보다 클 때)
                        if width < height:
                            continue
                        
                        # 새로운 줄 또는 문단 시작 여부 확인
                        new_line = False
                        new_paragraph = False
                        
                        if last_y is not None:
                            y_diff = abs(current_y - last_y)
                            
                            # 같은 줄인지 확인 (y좌표 차이가 임계값 이내)
                            if y_diff > y_threshold:
                                new_line = True
                                
                                # 문단 구분 확인 (줄 높이의 1.5배 이상 차이나면 새 문단으로 간주)
                                if y_diff > line_height * 1.5:
                                    new_paragraph = True
                            
                            # 같은 줄이지만 x좌표 간격이 너무 큰 경우 (들여쓰기 등) 새 줄로 간주
                            elif last_x_end is not None and (current_x_start - last_x_end) > x_gap_threshold:
                                new_line = True
                        
                        if new_line:
                            # 현재 줄 저장
                            if current_line:
                                lines.append(' '.join(current_line))
                                current_line = []
                            
                            # 문단 구분을 위한 빈 줄 추가
                            if new_paragraph:
                                lines.append('')
                        
                        current_line.append(text)
                        last_y = current_y
                        last_x_end = current_x_end
                    else:
                        current_line.append(text)
                
                # 마지막 줄 처리
                if current_line:
                    lines.append(' '.join(current_line))
                
                # 문장 및 문단 연결 처리
                processed_lines = []
                current_paragraph = []
                
                for i, line in enumerate(lines):
                    # 빈 줄은 문단 구분자로 처리
                    if not line.strip():
                        if current_paragraph:
                            # 문단 내 문장들을 연결하여 하나의 문단으로 만듦
                            processed_lines.append(' '.join(current_paragraph))
                            current_paragraph = []
                        processed_lines.append('')
                        continue
                    
                    # 문장 끝 판단 (마침표, 물음표, 느낌표로 끝나는 경우)
                    ends_with_sentence = line.rstrip().endswith(('.', '?', '!', '"', "'", '」', '』', '》', '）', '】', ')'))
                    
                    # 다음 줄이 대문자나 들여쓰기로 시작하는지 확인 (새 문장 시작 여부)
                    next_line_new_sentence = False
                    if i < len(lines) - 1 and lines[i+1].strip():
                        next_line = lines[i+1].strip()
                        if next_line[0].isupper() or next_line.startswith('  '):
                            next_line_new_sentence = True
                    
                    # 현재 줄을 문단에 추가
                    current_paragraph.append(line)
                    
                    # 문장이 끝나고 다음 줄이 새 문장이면 문단 완성
                    if ends_with_sentence and (next_line_new_sentence or i == len(lines) - 1):
                        processed_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                
                # 남은 문단 처리
                if current_paragraph:
                    processed_lines.append(' '.join(current_paragraph))
                
                # 마크다운 형식으로 구조화
                for i, line in enumerate(processed_lines):
                    if not line.strip():
                        markdown_text += "\n"
                        in_table = False
                        continue
                    
                    # 제목으로 보이는 텍스트는 마크다운 제목 형식으로 변환
                    if len(line) < 50 and (i == 0 or not processed_lines[i-1].strip()):
                        if i < len(processed_lines) - 1 and processed_lines[i+1].strip():
                            markdown_text += f"### {line}\n\n"
                            continue
                    
                    # 표로 보이는 텍스트는 마크다운 표 형식으로 변환
                    if "|" in line or "\t" in line:
                        cells = line.split("|") if "|" in line else line.split("\t")
                        if not in_table:
                            in_table = True
                            markdown_text += "| " + " | ".join(cells) + " |\n"
                            markdown_text += "| " + " | ".join(["---" for _ in cells]) + " |\n"
                        else:
                            markdown_text += "| " + " | ".join(cells) + " |\n"
                    else:
                        in_table = False
                        markdown_text += line + "\n\n"
                
            elif 'title' in result_json['images'][0]:
                title_text = result_json['images'][0].get('title',{}).get('inferText','')
                markdown_text = f"# {title_text}\n\n"

        if not markdown_text:
            return None, "이미지에서 텍스트를 추출하지 못했습니다."

        return markdown_text, None

    except requests.exceptions.RequestException as e:
        return None, f"OCR API 요청 실패: {e}"
    except json.JSONDecodeError:
        return None, f"OCR API 응답 파싱 실패: {response.text}"
    except Exception as e:
        return None, f"OCR 처리 중 예상치 못한 오류: {e}"

# --- 앱 UI 구성 ---
st.title("Koreanssam OCR")
st.write("PDF 또는 이미지 파일에서 텍스트를 추출하여 Markdown 형식으로 변환합니다.")

uploaded_file = st.file_uploader(
    "텍스트를 추출할 PDF 또는 이미지 파일을 업로드하세요",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "webp"]
)

select_pages = None
page_option = None
if uploaded_file is not None and uploaded_file.type == "application/pdf":
    page_option = st.radio(
        "처리할 페이지를 선택하세요:",
        ["모든 페이지", "특정 페이지"],
        key="page_option_radio"
    )
    if page_option == "특정 페이지":
        page_input = st.text_input("페이지 번호를 입력하세요 (예: 1,3,5)", key="page_input_text")
        if page_input:
            try:
                select_pages = [int(p.strip()) for p in page_input.split(",") if p.strip().isdigit()]
                if not select_pages:
                    st.warning("유효한 페이지 번호가 없습니다. 페이지 번호는 숫자로 입력해주세요.")
            except ValueError:
                st.error("숫자와 쉼표(,)만 사용하여 페이지 번호를 입력하세요.")

# --- 실행 로직 ---
if uploaded_file is not None and st.button("텍스트 추출 시작", key="start_button"):
    st.session_state.extracted_text = ""
    st.session_state.processing_done = False
    st.session_state.last_processed_type = None

    file_type = uploaded_file.type
    st.write(f"감지된 파일 타입: {file_type}")

    with st.spinner("텍스트 추출 중... 잠시만 기다려주세요."):
        if file_type == "application/pdf":
            st.session_state.last_processed_type = 'pdf'
            
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            try:
                # PDF 파일 저장
                pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # OCR 처리
                processor = ClovaOCRProcessor(NAVER_OCR_API_URL, NAVER_OCR_SECRET_KEY)
                extracted_text = processor.process_pdf(pdf_path)
                
                st.session_state.extracted_text = extracted_text
                st.session_state.processing_done = True
                st.success("PDF 텍스트 추출 완료!")
            except Exception as e:
                st.error(f"PDF 처리 중 오류 발생: {str(e)}")
                import traceback
                st.text_area("Traceback (PDF)", traceback.format_exc(), height=150)
            finally:
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir, ignore_errors=True)

        elif file_type.startswith("image/"):
            st.session_state.last_processed_type = 'image'
            try:
                img_bytes = uploaded_file.getvalue()
                img_format = file_type.split('/')[-1]
                markdown_text, error_msg = call_naver_ocr_image(img_bytes, img_format)
                
                if error_msg:
                    st.error(f"❌ OCR 오류: {error_msg}")
                else:
                    st.session_state.extracted_text = markdown_text
                    st.session_state.processing_done = True
                    st.success("✅ 이미지 텍스트 추출 완료!")
            except Exception as e:
                st.error(f"이미지 처리 중 오류 발생: {str(e)}")
                import traceback
                st.text_area("Traceback (Image)", traceback.format_exc(), height=150)
        else:
            st.error(f"지원하지 않는 파일 타입입니다: {file_type}. PDF 또는 이미지를 업로드해주세요.")

# --- 결과 표시 및 상호작용 ---
if st.session_state.processing_done and st.session_state.extracted_text:
    st.markdown("---")
    st.subheader("📄 추출된 텍스트 결과")

    # 코드 블록으로 표시
    st.code(st.session_state.extracted_text, language="markdown", line_numbers=False)

    # 다운로드 버튼
    download_filename = "extracted_text.md"
    mime_type = "text/markdown"
    if uploaded_file:
        original_filename_stem = Path(uploaded_file.name).stem
        download_filename = f"{original_filename_stem}_extracted.md"

    st.download_button(
        label="💾 결과 다운로드",
        data=st.session_state.extracted_text,
        file_name=download_filename,
        mime=mime_type,
        key="download_button"
    )

    st.markdown("---")
