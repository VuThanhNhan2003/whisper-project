from fileinput import filename
import os
import json
import io
import time
import uuid
import re
import socket
import unicodedata
import ssl
import torch
import fitz  # PyMuPDF
import sys
import gc
import time
import threading
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= CẤU HÌNH =================
#FOLDER_ID = '1JRP0iOIHqYE7VBUhGClAnUkOMbC3Uy1m'
FOLDER_ID = os.environ.get('DRIVE_FOLDER_ID', '1VoiAS75PpupVHPUq64rfU_4guclIM36s')
#FOLDER_ID = '1VoiAS75PpupVHPUq64rfU_4guclIM36s'
CREDENTIALS_FILE = 'credentials.json'
OUTPUT_FILE = 'KNOWLEDGE_BASE_QWEN.json'
#LOCAL_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
#LOCAL_MODEL_PATH = "./models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
#LOCAL_MODEL_PATH = "./models/Qwen3-30B"
LOCAL_MODEL_PATH = "./models/Qwen3-VL-8B-Instruct"
# --- FIX LỖI MẠNG ---
socket.setdefaulttimeout(600)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass
model = None
processor = None
# ================= 1. LOAD MODEL =================
print(f"⏳ Đang khởi động Qwen 3.0 VL từ {LOCAL_MODEL_PATH}...", flush=True)

try:
    # TẠO LUỒNG BÁO CÁO TIẾN TRÌNH LOAD MODEL
    loading_stop_event = threading.Event()
    
    def print_loading_status():
        steps = [
            "Đang quét file weights cục bộ trên ổ đĩa mạng (NFS)...",
            "Đang nạp dữ liệu và phân bổ vào RAM hệ thống...",
            "Đang đẩy hàng tỷ tham số (Parameters) lên VRAM của GPU...",
            "Đang biên dịch các lớp Attention (SDPA) và ép kiểu bfloat16...",
            "Đang khởi tạo bộ xử lý hình ảnh Vision Processor..."
        ]
        step_idx = 0
        while not loading_stop_event.is_set():
            if step_idx < len(steps):
                print(f"   ⚙️ [Khởi động] {steps[step_idx]}", flush=True)
                step_idx += 1
            else:
                print(f"   ⚙️ [Khởi động] Đang hoàn tất liên kết GPU, sắp xong rồi...", flush=True)
            
            loading_stop_event.wait(8.0)

    loading_thread = threading.Thread(target=print_loading_status, daemon=True)
    loading_thread.start()

    try:
        # --- LOAD MODEL ---
        model = AutoModelForImageTextToText.from_pretrained(
            LOCAL_MODEL_PATH,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,  
            # --- CẤU HÌNH TỐI ƯU CHO A100 ---
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            ignore_mismatched_sizes=True
        )
        #model = torch.compile(model)
        # --- LOAD PROCESSOR 
        processor = AutoProcessor.from_pretrained(
            LOCAL_MODEL_PATH, 
            local_files_only=True,
            trust_remote_code=True,
            min_pixels=256*28*28, 
            max_pixels=1024*28*28   # <--- Sửa thành 1024 cho nhanh mà vẫn nét
        )
        
    finally:
        # CỰC KỲ QUAN TRỌNG: Tắt luồng báo cáo khi load xong
        loading_stop_event.set()
        loading_thread.join()

    print(f"✅ Model OK! Sẵn sàng trên: {model.device}\n", flush=True)

except Exception as e:
    print(f"❌ Lỗi Load Model: {e}", flush=True)
    # Gợi ý fix nếu vẫn lỗi
    print("💡 Gợi ý: Hãy thử cài bản transformers mới nhất từ GitHub bằng lệnh:", flush=True)
    print("   pip install git+https://github.com/huggingface/transformers", flush=True)
    exit()

# ================= 2. DRIVE DOWNLOADER =================
def get_drive_service():
    if not os.path.exists(CREDENTIALS_FILE): return None
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_files(service, folder_id):
    query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def clean_vn_string(s):
    # Chuẩn hóa về dạng NFC để so khớp tiếng Việt chính xác
    return unicodedata.normalize('NFC', s.lower())

def find_subject_recursive(service, folder_id):
  
    current_id = folder_id
    
    # Các từ khóa để nhận diện Folder môn học
    subject_keywords = [clean_vn_string(k) for k in["Môn", "môn", "course", "subject", "học phần", "class", "lớp"]]
    
    # Các từ khóa để nhận diện Folder rác (cần leo tiếp)
    ignore_keywords = [clean_vn_string(k) for k in ["slide", "bài giảng", "tài liệu", "pdf", "file", "folder", "chương", "week", "tuần", "files", "slides", "document", "docs", "tài liệu", "tài liệu tham khảo", "references", "đề cương", "syllabus", "lab", "thực hành", "assignment", "bài tập", "exercise", "đề thi", "exam", "quiz", "video"]]

    for i in range(6): # Leo tối đa 6 cấp
        try:
            file = service.files().get(fileId=current_id, fields='id, name, parents', supportsAllDrives=True).execute()
            name = file.get('name', '')
            parents = file.get('parents', [])
            
            name_lower = name.lower()
            
            # 1. Nếu tên folder chứa từ khóa môn học -> LẤY LUÔN
            if any(k in name_lower for k in subject_keywords):
                return name
            
            # 2. Nếu tên folder KHÔNG chứa từ khóa "rác" (như slide, pdf...) -> Có thể là tên môn -> LẤY LUÔN
            if not any(k in name_lower for k in ignore_keywords):
                return name

            # 3. Nếu không phải 2 trường hợp trên, leo tiếp lên cha
            if parents:
                current_id = parents[0]
            else:
                # Đã lên tới đỉnh (Root) mà chưa tìm thấy -> Đành lấy tên hiện tại
                return name
        except:
            # Lỗi (do hết quyền truy cập hoặc hết path) -> Dừng
            break
            
            
    # Nếu lỗi hoặc không tìm ra, trả về "General Subject"
    return "General Subject"

def download_and_convert_file(service, file_info):
    file_id = file_info['id']; name = file_info['name']; mime = file_info['mimeType']

    clean_name = re.sub(r'[\\/*?:"<>|]', "", os.path.splitext(name)[0]) + ".pdf"
    if os.path.exists(clean_name): return clean_name

    print(f"⬇️ Đang xử lý: {name}...", flush=True)
    request = None

    # 1. Google Docs/Slides -> Export PDF (Luôn thành công)
    if 'application/vnd.google-apps' in mime:
        request = service.files().export_media(fileId=file_id, mimeType='application/pdf')

    # 2. File Word/PPT upload lên -> Thử Export PDF (Hên xui tùy Google API)
    elif 'wordprocessingml' in mime or 'presentationml' in mime:
        try:
            request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
        except:
            print(f"   ⚠️ Không thể convert '{name}' trên Server Linux. Hãy up file PDF!", flush=True)
            return None

    # 3. File PDF gốc -> Tải về
    elif 'application/pdf' in mime:
        request = service.files().get_media(fileId=file_id)

    else:
        return None # File lạ bỏ qua

    if request:
        try:
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done: _, done = downloader.next_chunk()
            with open(clean_name, 'wb') as f: f.write(fh.getbuffer())
            return clean_name
        except Exception as e:
            print(f"   ❌ Lỗi tải: {e}", flush=True)
            return None
    return None

# ================= 3. LOGIC XỬ LÝ (PROMPT MỚI) =================
def repair_json(text):
    try:
        # 1. Làm sạch markdown
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # 2. Tìm điểm bắt đầu JSON
        start = text.find('[')
        if start == -1: return text # Không tìm thấy JSON
        text = text[start:]

        # 3. XỬ LÝ CẮT CỤT (TRUNCATION) - QUAN TRỌNG NHẤT
        # Nếu không kết thúc bằng ']', tức là bị cắt giữa chừng
        if not text.endswith(']'):
            # Xóa các dấu phẩy hoặc khoảng trắng thừa ở cuối
            text = text.rstrip(', \n\r\t')
            
            # Đếm xem đang mở ngoặc kép hay không để đóng lại
            if text.count('"') % 2 != 0:
                text += '"'
            
            # Đếm ngoặc nhọn { } để đóng object
            open_braces = text.count('{')
            close_braces = text.count('}')
            if open_braces > close_braces:
                text += '}'
            
            # Cuối cùng đóng mảng
            text += ']'
            
        return text
    except:
        return text

def process_pdf_with_qwen(pdf_path, filename, subject_name):

    global model, processor

    extracted_items = []

    try:

        doc = fitz.open(pdf_path)
        total_pages = len(doc) # <--- THÊM: Lấy tổng số trang
        print(f"🧠 Đang đọc '{filename}' ({total_pages} trang)...", flush=True)
        pbar = tqdm(total=total_pages, desc="Tiến độ", unit="trang", file=sys.stdout, mininterval=2.0)
        
        for i, page in enumerate(doc):

            tqdm.write(f"\n\n 👁️ Trang {i+1}...")

            pix = page.get_pixmap(dpi=200)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)



            # --- PROMPT CHI TIẾT CỦA BẠN ---

            prompt_text = f"""

    Bạn là chuyên gia Xử lý dữ liệu RAG, nhiệm vụ cung cấp kiến thức cho chatbot AI.

    Nhiệm vụ: Chuyển đổi nội dung hình ảnh trang tài liệu thành các "TextNode" JSON.


    INPUT: Hình ảnh trang tài liệu (Chứa text, công thức, code, ảnh).



    YÊU CẦU XỬ LÝ:

    1. **PHÂN ĐOẠN (SEGMENTATION)**: Chia văn bản thành các Node dựa trên MẠCH KIẾN THỨC.

    2. **NỘI DUNG (TEXT FIELD)**:
       - Bỏ qua logo trường học, tên giảng viên nếu có (Vd: Ths. Nguyễn Văn A, Ths. Trần Thị B, Thầy Nguyễn Văn C, Cô Trần Thị D, PGS.TS Nguyễn Văn E).
       - Loại bỏ các nội dung ở footer và header (nếu có, thường là tên trường hay quảng cáo..., giới thiệu, chứ không phải là tiêu đề nội dung slide), thông tin hành chính không liên quan, số trang, hoặc các câu mang tính chất chuyển ý không có giá trị học thuật (VD: 'Nội dung bài học hôm nay gồm', 'Cảm ơn các bạn')."
       - Giữ lại toàn bộ định nghĩa, giải thích chi tiết, công thức toán học.
    3. **XỬ LÝ TOÁN HỌC & LATEX**:
        - MỌI biểu thức toán học, công thức, biến số (x, y, i, j), ma trận, tích phân, đạo hàm, phân số... BẮT BUỘC phải chuyển sang mã chuẩn LaTeX.
        - TUYỆT ĐỐI KHÔNG dùng ký tự Unicode toán học (như α, β, ∑, ∫, √). Phải dùng \\alpha, \\beta, \\sum, \\int, \\sqrt.
        - Công thức nằm CÙNG DÒNG với chữ (inline): Bọc trong cặp dấu `$`. (Ví dụ: Cho biến $x$ và hàm $f(x) = x^2$).
        - Công thức ĐỨNG RIÊNG 1 DÒNG (block): Bọc trong cặp dấu `$$`. (Ví dụ: $$\\lim_{{x \\to \\infty}} f(x)$$).
        - KHÔNG dùng `\\(` `\\)` hay `\\[` `\\]`. Chỉ dùng `$` và `$$`.
        - Escape đúng các dấu backslash trong JSON (ví dụ: `\\\\frac{{a}}{{b}}`).
        + Lớn hơn hoặc bằng: dùng `\\le`. Bé hơn hoặc bằng: dùng `\\ge`. Khác: dùng `\\neq`.
          + Tương đương: dùng `\\Leftrightarrow`. Suy ra: dùng `\\Rightarrow`.
          + Với mọi: dùng `\\forall`. Tồn tại: dùng `\\exists`.
        Với các bài giải có nhiều bước biến đổi dấu bằng, bắt buộc dùng môi trường Aligned để căn lề cho đẹp: 
            `$$\begin{"aligned"} vế_trái &= vế_phải \\ &= kết_quả \end{"aligned"}$$
    - **Code**: Đặt trong block markdown (ví dụ: ```java ... ```) bên trong trường `text`.

    - **OCR CODE**: Nếu thấy ảnh chụp màn hình chứa code, PHẢI gõ lại chính xác từng dòng code đó.

    - **DIAGRAMS**: Nếu có biểu đồ, hãy mô tả logic của nó bằng lời văn.

    - **KHÔNG TÓM TẮT**: Giữ nguyên nội dung gốc của slide, viết cho đầy đủ tránh bị khác biệt so với văn bản gốc.

    5. **METADATA**: Trích xuất ngữ cảnh.

    6. **KHÔNG MÔ TẢ CHUNG CHUNG**: Cấm viết kiểu "Trang này nói về...", "Slide này trình bày...".

    OUTPUT JSON FORMAT (Bắt buộc đúng cấu trúc này, slide tiếng Việt thì viết tiếng Việt, tiếng anh thì phải viết tiếng Anh):

    8. **XỬ LÝ DẤU CÂU (QUAN TRỌNG)**: Nếu trong văn bản gốc có dấu ngoặc kép ("), hãy đổi thành dấu ngoặc đơn (') để tránh lỗi JSON. Ví dụ: "Deadlock" -> 'Deadlock'.

    [

      {{

        "text": "Viết ra nội dung của slide cách chính xác,  KHÔNG PHẢI TRANG NÀY CÓ GÌ (nội dung chi tiết, tránh để khác biệt so với nội dung gốc, không tóm tắt, nếu có code thì để trong block markdown, nếu có công thức toán học thì đổi thành định dạng latex như quy định ở trên, nếu có biểu đồ thì mô tả chi tiết biểu đồ đó bằng lời văn)",

        "metadata": {{

            "subject": {subject_name},

            "page": {i+1},

            "topic": "Tên chủ đề cụ thể",

            "category": "Algorithm/Theory/Example",

            "keywords": ["key1", "key2"],

            "has_code": true

        }}

      }}

    ]

    CHỈ TRẢ VỀ JSON HỢP LỆ.

            """



            messages = [{

                "role": "user",

                "content": [

                    {"type": "image", "image": img},

                    {"type": "text", "text": prompt_text},

                ],

            }]



            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 1. Lấy cả video_inputs

            image_inputs, video_inputs = process_vision_info(messages)

            # 2. Truyền videos=video_inputs vào (QUAN TRỌNG)

            inputs = processor(

                text=[text_input],

                images=image_inputs,

                videos=video_inputs,

                padding=True,

                return_tensors="pt"

            ).to(model.device)

            stop_event = threading.Event()
            
            def print_status_steps():
                steps = [
                    "Đang nạp dữ liệu ảnh vào GPU VRAM...",
                    "Đang phân tích OCR và nhận diện cấu trúc...",
                    "Đang chuyển đổi mã Toán học (LaTeX) và sinh văn bản...",
                    "Đang hoàn thiện định dạng chuỗi JSON..."
                ]
                step_idx = 0
                while not stop_event.is_set():
                    if step_idx < len(steps):
                        tqdm.write(f"   ⏳ [Đang xử lý] {steps[step_idx]}")
                        step_idx += 1
                    else:
                        tqdm.write(f"   ⏳ [Đang xử lý] AI đang tối ưu hóa chuỗi JSON...")

                    # Chờ 6s trước khi báo dòng tiếp theo. Dừng ngay nếu AI đã xong.
                    stop_event.wait(6.0)

            status_thread = threading.Thread(target=print_status_steps, daemon=True)
            status_thread.start()

            try:
                # --- BẮT ĐẦU CHO AI SUY NGHĨ ---
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=6144, do_sample=False,use_cache=True)
            
            finally:
                stop_event.set()
                status_thread.join()

            # Cắt bỏ phần input_ids để chỉ lấy nội dung AI mới sinh ra
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]



            try:

                clean_json = repair_json(output_text)

                nodes = json.loads(clean_json)

                for node in nodes:

                    node['id_'] = str(uuid.uuid4())

                    node['metadata']['file_name'] = filename

                    node['metadata']['subject'] = subject_name

                    # Code tự điền page nếu AI quên

                    node['metadata']['page'] = i + 1



                extracted_items.extend(nodes)

                subj = nodes[0]['metadata'].get('subject', 'Unknown')

                tqdm.write(f"      ✅ Trang {i+1}: OK (Môn: {subj})")

            except:

                tqdm.write(f"      ⚠️ Trang {i+1}: Lỗi parse JSON.")

            del inputs, generated_ids, image_inputs, video_inputs

            torch.cuda.empty_cache()

            gc.collect()
            pbar.update(1)
        pbar.close()
        doc.close()

    except Exception as e:

        print(f"❌ Lỗi file {filename}: {e}", flush=True)

    return extracted_items
# ================= 4. MAIN =================
def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    service = get_drive_service()
    if not service: return

    # 1. Tìm tên môn học (Dùng hàm CŨ của bạn)
    print("🕵️ Đang xác định tên môn học...", flush=True)
    subject_name = find_subject_recursive(service, FOLDER_ID)
    print(f"✅ Môn học: {subject_name}", flush=True)

    # 2. Lấy thông tin folder hiện tại để làm hậu tố tên file (VD: Slides)
    try:
        folder_info = service.files().get(fileId=FOLDER_ID, fields='name').execute()
        current_folder_name = folder_info.get('name', 'General')
    except:
        current_folder_name = 'General'

    # Tạo tên file JSON: [TenMon]_[TenFolderCon].json
    # Xử lý tên để bỏ dấu tiếng Việt và ký tự lạ (để tránh lỗi file system)
    clean_subject = re.sub(r'[^\w\s-]', '', subject_name).strip().lower().replace(' ', '')
    clean_folder = re.sub(r'[^\w\s-]', '', current_folder_name).strip().lower().replace(' ', '')
    output_filename = f"{clean_subject}_{clean_folder}.json"
    
    print(f"💾 File JSON sẽ được lưu với tên: {output_filename}", flush=True)

    # 3. Lấy danh sách file và xử lý
    print(f"\n☁️ Lấy danh sách file từ folder: {current_folder_name}...", flush=True)
    drive_files = list_files(service, FOLDER_ID)

    final_kb = []
    # Load lại data cũ (Resume)
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f: final_kb = json.load(f)
            print(f"📚 Đã load {len(final_kb)} nodes cũ.", flush=True)
        except: pass

    print("\n--- BẮT ĐẦU XỬ LÝ (LOCAL GPU) ---")
    # Tạo một set chứa tên các file đã làm để tra cứu cho nhanh
    processed_files = set()
    for node in final_kb:
        if 'file_name' in node.get('metadata', {}):
            processed_files.add(node['metadata']['file_name'])

    print("\n--- BẮT ĐẦU XỬ LÝ (LOCAL GPU) ---")
    for file_info in drive_files:
        # --- CHECK TRÙNG (RESUME LOGIC) ---
        if file_info['name'] in processed_files:
            print(f"⏩ Đã xong, bỏ qua: {file_info['name']}", flush=True)
            continue
        # ----------------------------------

        try:
            pdf_path = download_and_convert_file(service, file_info)
            if pdf_path:
                new_nodes = process_pdf_with_qwen(pdf_path, file_info['name'], subject_name)
                if new_nodes:
                    final_kb.extend(new_nodes)
                    # Cập nhật luôn vào danh sách đã làm
                    processed_files.add(file_info['name'])
                    
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(final_kb, f, ensure_ascii=False, indent=2)
                    print(f"💾 Đã lưu: {file_info['name']} (+{len(new_nodes)} nodes) vào {output_filename}", flush=True)
        except Exception as e:
            print(f"❌ Lỗi file {file_info['name']}: {e}")

    print("🏆 HOÀN TẤT!", flush=True)

if __name__ == '__main__':
    main()