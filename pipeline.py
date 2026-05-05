import os
import re
import subprocess
import tempfile
import json
import threading
import time
from datetime import datetime
from queue import Full, Queue
from urllib.parse import urlparse
from typing import Dict, Optional

import requests
from faster_whisper import WhisperModel
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from schemas import JobStatus, TranscriptionOptions


jobs_status: Dict[str, JobStatus] = {}
_drive_service = None
_drive_file_name_cache: Dict[str, str] = {}
_jobs_store_path = os.getenv("JOBS_STORE_PATH", "file_vtt/.jobs_status.json")
_job_payload_store_path = os.getenv("JOB_PAYLOAD_STORE_PATH", "file_vtt/.job_payloads.json")
_job_payloads: Dict[str, dict] = {}
_job_queue_maxsize = max(100, int(os.getenv("JOB_QUEUE_MAXSIZE", "2000")))
_job_queue: Queue = Queue(maxsize=_job_queue_maxsize)
_worker_threads: list[threading.Thread] = []
_worker_lock = threading.Lock()
_jobs_status_lock = threading.Lock()
_thread_local = threading.local()
_UNSET = object()
_recovery_completed = False
_download_max_retries = max(1, int(os.getenv("DOWNLOAD_MAX_RETRIES", "4")))
_download_retry_delay_seconds = max(1, int(os.getenv("DOWNLOAD_RETRY_DELAY_SECONDS", "2")))
_direct_download_connect_timeout = max(5, int(os.getenv("DIRECT_DOWNLOAD_CONNECT_TIMEOUT", "20")))
_direct_download_read_timeout = max(30, int(os.getenv("DIRECT_DOWNLOAD_READ_TIMEOUT", "300")))
_google_api_retries = max(1, int(os.getenv("GOOGLE_API_RETRIES", "5")))


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(_download_retry_delay_seconds * attempt)


def _ensure_jobs_store_parent() -> None:
    parent_dir = os.path.dirname(_jobs_store_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _persist_jobs_status() -> None:
    _ensure_jobs_store_parent()
    payload = {job_id: status.model_dump() for job_id, status in jobs_status.items()}
    tmp_path = f"{_jobs_store_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, _jobs_store_path)


def _persist_job_payloads() -> None:
    _ensure_jobs_store_parent()
    tmp_path = f"{_job_payload_store_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_job_payloads, f, ensure_ascii=False)
    os.replace(tmp_path, _job_payload_store_path)


def _load_jobs_status() -> None:
    if not os.path.exists(_jobs_store_path):
        return

    try:
        with open(_jobs_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return

        for job_id, raw in data.items():
            try:
                status = JobStatus(**raw)
                if status.status in {"queued", "processing"}:
                    status.status = "failed"
                    status.progress = "Interrupted by server restart"
                    if not status.error:
                        status.error = "Server restarted before job completion"
                jobs_status[job_id] = status
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Failed to load jobs status store: {e}")


def _load_job_payloads() -> None:
    if not os.path.exists(_job_payload_store_path):
        return

    try:
        with open(_job_payload_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _job_payloads.update(data)
    except Exception as e:
        print(f"⚠️ Failed to load job payload store: {e}")


def initialize_job_store() -> None:
    with _jobs_status_lock:
        if jobs_status:
            return
        _load_job_payloads()
        _load_jobs_status()
        if jobs_status or _job_payloads:
            _persist_jobs_status()
            _persist_job_payloads()


def create_job_status(job_id: str, status: JobStatus) -> None:
    with _jobs_status_lock:
        jobs_status[job_id] = status
        _persist_jobs_status()


def register_job_payload(job_id: str, video_url: str, language: str, request: TranscriptionOptions) -> None:
    with _jobs_status_lock:
        options_payload = TranscriptionOptions(**request.model_dump()).model_dump()
        _job_payloads[job_id] = {
            "video_url": video_url,
            "language": language,
            "request": options_payload,
        }
        _persist_job_payloads()


def remove_job_payload(job_id: str) -> None:
    with _jobs_status_lock:
        if job_id in _job_payloads:
            del _job_payloads[job_id]
            _persist_job_payloads()


def update_job_status(
    job_id: str,
    *,
    status: object = _UNSET,
    progress: object = _UNSET,
    error: object = _UNSET,
    result: object = _UNSET,
) -> None:
    with _jobs_status_lock:
        current = jobs_status.get(job_id)
        if current is None:
            return

        if status is not _UNSET:
            current.status = status
        if progress is not _UNSET:
            current.progress = progress
        if error is not _UNSET:
            current.error = error
        if result is not _UNSET:
            current.result = result

        _persist_jobs_status()


def remove_job_status(job_id: str) -> bool:
    with _jobs_status_lock:
        if job_id not in jobs_status:
            return False
        del jobs_status[job_id]
        _persist_jobs_status()
        if job_id in _job_payloads:
            del _job_payloads[job_id]
            _persist_job_payloads()
        return True


def _recover_jobs_after_restart() -> None:
    global _recovery_completed

    with _jobs_status_lock:
        if _recovery_completed:
            return

        recovered_items = []
        for job_id, status in jobs_status.items():
            if status.status not in {"queued", "processing"}:
                continue

            payload = _job_payloads.get(job_id)
            if not payload:
                status.status = "failed"
                status.progress = "Interrupted by server restart"
                if not status.error:
                    status.error = "Server restarted before job completion"
                continue

            try:
                request_obj = TranscriptionOptions(**payload.get("request", {}))
                recovered_items.append(
                    (
                        job_id,
                        payload.get("video_url", ""),
                        payload.get("language", "vi"),
                        request_obj,
                    )
                )
                status.status = "queued"
                status.progress = "Recovered after restart - queued again"
                status.error = None
            except Exception as e:
                status.status = "failed"
                status.progress = "Interrupted by server restart"
                status.error = f"Failed to recover job payload: {e}"

        _persist_jobs_status()
        _recovery_completed = True

    recovered_count = 0
    for job_id, video_url, language, request_obj in recovered_items:
        try:
            _job_queue.put_nowait((job_id, video_url, language, request_obj))
            recovered_count += 1
            update_job_status(job_id, progress="Recovered after restart - queued in worker")
        except Full:
            update_job_status(
                job_id,
                status="failed",
                progress="Recovery failed: queue full",
                error=f"Job queue is full ({_job_queue_maxsize}) while recovering",
            )

    if recovered_count:
        print(f"♻️ Recovered {recovered_count} interrupted job(s) after restart")


def _default_worker_count() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(2, cpu // 2 if cpu > 1 else 1))


def get_worker_count() -> int:
    raw = os.getenv("TRANSCRIPTION_WORKERS")
    if not raw:
        return _default_worker_count()

    try:
        return max(1, int(raw))
    except ValueError:
        return _default_worker_count()


def get_whisper_num_workers() -> int:
    raw = os.getenv("WHISPER_NUM_WORKERS")
    if not raw:
        return 2

    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def start_job_workers() -> None:
    initialize_job_store()

    with _worker_lock:
        if _worker_threads:
            return

        worker_count = get_worker_count()
        for i in range(worker_count):
            t = threading.Thread(
                target=_worker_loop,
                name=f"transcription-worker-{i + 1}",
                daemon=True,
            )
            t.start()
            _worker_threads.append(t)

        print(f"✅ Started {len(_worker_threads)} transcription worker(s)")

    _recover_jobs_after_restart()


def enqueue_transcription_job(job_id: str, video_url: str, language: str, request: TranscriptionOptions) -> None:
    start_job_workers()
    register_job_payload(job_id, video_url, language, request)

    try:
        _job_queue.put_nowait((job_id, video_url, language, request))
        position = _job_queue.qsize()
        if job_id in jobs_status and jobs_status[job_id].status == "queued":
            update_job_status(job_id, progress=f"Queued in worker queue (position approx: {position})")
    except Full:
        remove_job_payload(job_id)
        raise Exception(
            f"Job queue is full ({_job_queue_maxsize}). Try again later or increase JOB_QUEUE_MAXSIZE"
        )


def get_processing_metrics() -> dict:
    processing = len([j for j in jobs_status.values() if j.status == "processing"])
    queued = len([j for j in jobs_status.values() if j.status == "queued"])
    return {
        "worker_count": len(_worker_threads),
        "queue_size": _job_queue.qsize(),
        "queue_capacity": _job_queue_maxsize,
        "processing_jobs": processing,
        "queued_jobs": queued,
    }


def _get_thread_model(model_name: str) -> WhisperModel:
    cache: Optional[dict] = getattr(_thread_local, "model_cache", None)
    if cache is None:
        cache = {}
        _thread_local.model_cache = cache

    model = cache.get(model_name)
    if model is None:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            num_workers=get_whisper_num_workers(),
        )
        cache[model_name] = model

    return model


def _worker_loop() -> None:
    while True:
        job_id, video_url, language, request = _job_queue.get()
        try:
            process_video_transcription(job_id, video_url, language, request)
        except Exception as e:
            if job_id in jobs_status:
                update_job_status(job_id, status="failed", error=str(e))
            remove_job_payload(job_id)
            print(f"❌ Worker failed for job {job_id}: {e}")
        finally:
            _job_queue.task_done()


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", name).strip()
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized or f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_drive_api_key(credentials_path: str = "credentials.json") -> str | None:
    if not os.path.exists(credentials_path):
        return os.getenv("GOOGLE_DRIVE_API_KEY")

    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("api_key") or data.get("key") or os.getenv("GOOGLE_DRIVE_API_KEY")
    except Exception:
        return os.getenv("GOOGLE_DRIVE_API_KEY")


def get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    credentials_path = "credentials.json"
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("type") == "service_account":
                creds = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                )
                _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
                return _drive_service
        except Exception:
            pass

    api_key = _load_drive_api_key(credentials_path)
    if api_key:
        _drive_service = build("drive", "v3", developerKey=api_key, cache_discovery=False)
        return _drive_service

    raise Exception(
        "Google Drive credentials not found. Provide credentials.json (service account or api_key) "
        "or set GOOGLE_DRIVE_API_KEY"
    )


def is_google_drive_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in {"drive.google.com", "www.drive.google.com"} or url.startswith("gdrive://")


def parse_google_drive_url(url: str) -> tuple[str, str]:
    if url.startswith("gdrive://file/"):
        return "file", url.split("gdrive://file/", 1)[1]

    if "/folders/" in url:
        folder_id = url.split("/folders/", 1)[1].split("?", 1)[0].split("/", 1)[0]
        if folder_id:
            return "folder", folder_id

    if "/file/d/" in url:
        file_id = url.split("/file/d/", 1)[1].split("/", 1)[0].split("?", 1)[0]
        if file_id:
            return "file", file_id

    parsed = urlparse(url)
    query = parsed.query or ""
    if "id=" in query:
        file_id = query.split("id=", 1)[1].split("&", 1)[0]
        if file_id:
            return "file", file_id

    raise Exception("Unsupported Google Drive URL format")


def list_google_drive_video_files(folder_url: str) -> list[dict[str, str]]:
    url_type, folder_id = parse_google_drive_url(folder_url)
    if url_type != "folder":
        raise Exception("Expected a Google Drive folder URL")

    service = get_drive_service()
    page_token = None
    results = []

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false and mimeType contains 'video/'",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=100,
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute(num_retries=_google_api_retries)

        for file in response.get("files", []):
            results.append(
                {
                    "id": file["id"],
                    "name": file["name"],
                    "mimeType": file.get("mimeType", ""),
                }
            )

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results


def download_google_drive_file(file_id: str) -> str:
    last_error = None

    for attempt in range(1, _download_max_retries + 1):
        temp_file_path = None
        try:
            service = get_drive_service()
            metadata = service.files().get(
                fileId=file_id,
                fields="id,name,mimeType",
                supportsAllDrives=True,
            ).execute(num_retries=_google_api_retries)

            file_name = metadata.get("name", f"drive_{file_id}")
            _drive_file_name_cache[file_id] = file_name
            suffix = os.path.splitext(file_name)[1] or ".mp4"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.close()
            temp_file_path = temp_file.name

            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
            with open(temp_file_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=_google_api_retries)

            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                raise Exception("Downloaded Google Drive file is empty")

            return temp_file_path

        except Exception as e:
            last_error = e
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if attempt < _download_max_retries:
                print(
                    f"⚠️ Google Drive download attempt {attempt}/{_download_max_retries} failed for {file_id}: {e}. Retrying..."
                )
                _sleep_before_retry(attempt)
                continue
            break

    raise Exception(
        f"Google Drive download failed after {_download_max_retries} attempts: {last_error}"
    )


def detect_hallucination_patterns(text: str) -> bool:
    """Phát hiện các pattern hallucination phổ biến"""
    if re.search(r'(.)\1{10,}', text):
        return True

    words = text.split()
    if len(words) > 5:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > len(words) * 0.3:
                return True

    spam_patterns = [
        r'subscribe.*kênh',
        r'đăng ký.*kênh',
        r'like.*share',
        r'ghiền.*gõ',
        r'không.*bỏ.*lỡ',
        r'video.*hấp.*dẫn'
    ]

    text_lower = text.lower()
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def clean_repetitive_text(text: str) -> str:
    """Làm sạch text bị lặp"""
    text = re.sub(r'(.)\1{5,}', r'\1', text)

    words = text.split()
    cleaned_words = []
    prev_word = ""
    repeat_count = 0

    for word in words:
        if word.lower() == prev_word.lower():
            repeat_count += 1
            if repeat_count < 2:
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            repeat_count = 0
        prev_word = word

    return ' '.join(cleaned_words)


def validate_segment_quality(segment, min_duration=0.5, max_duration=30.0) -> bool:
    """Kiểm tra chất lượng segment"""
    duration = segment.end - segment.start

    if duration < min_duration or duration > max_duration:
        return False

    if duration > 8 and len(segment.text.strip()) < 10:
        return False

    if detect_hallucination_patterns(segment.text):
        return False

    return True


def format_timestamp(seconds, vtt=False):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    sep = '.' if vtt else ','
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millisecs:03d}"


def get_default_prompt_template(language: str = "vi") -> str:
    """Default prompt template cho LLM refinement - support mixed lang, code, LaTeX"""
    templates = {
        "vi": """Bạn là một trợ lý chỉnh sửa phiên bản và cải thiện chất lượng transcript từ Whisper.

HƯỚNG DẪN CHÍNH:
1. FIX ONLY: Chính tả, diacritics tiếng Việt, viết hoa đúng chủng loại, ngữ pháp nhỏ
2. PRESERVE: Code, công thức LaTeX, abbreviations, technical terms tiếng Anh, cấu trúc gốc
3. LENGTH: Giữ mỗi segments trong khoảng 50-70 ký tự. Không thêm, không bớt nội dung
4. CONTEXT: Output phải giữ nguyên ý nghĩa, tính chính xác khoa học, thứ tự từ

KHÔNG LÀM:
- Không dịch tiếng Anh sang tiếng Việt
- Không sửa code hay công thức toán
- Không thêm giải thích hoặc comment bổ sung
- Không thay đổi timestamps (sẽ được xử lý riêng)

OUTPUT: Chỉ trả về JSON array thuần (không markdown, không giải thích) với cùng số lượng items input.
Mỗi item: {{"id": int, "text": "refined text"}}
""",
        "en": """You are a transcript quality refinement assistant from Whisper transcription.

MAIN INSTRUCTIONS:
1. FIX ONLY: Spelling, capitalization, minor grammar, punctuation
2. PRESERVE: Code, LaTeX formulas, technical terms, abbreviations, original structure
3. LENGTH: Keep each segment within 50-70 characters. Do not add or remove content
4. CONTEXT: Output must maintain meaning, technical accuracy, original order

DO NOT:
- Translate between languages
- Fix code or mathematical formulas
- Add explanations or comments
- Modify timestamps (handled separately)

OUTPUT: Return JSON array only (no markdown, no explanation) with same number of items as input.
Each item: {{"id": int, "text": "refined text"}}
"""
    }
    return templates.get(language, templates["en"])


def _extract_json_array(text: str):
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def _resolve_llm_urls(llm_proxy_url: str) -> list[str]:
    base = llm_proxy_url.rstrip("/")
    if base.endswith("/v1/chat/completions") or base.endswith("/chat/completions"):
        return [base]
    return [
        f"{base}/v1/chat/completions",
        f"{base}/chat/completions",
        f"{base}/chat",
    ]


# =============================================================================
# RAG Context Query
# =============================================================================

def query_rag_context(
    text_window: str,
    subject: Optional[str],
    rag_api_url: str,
    timeout: int = 20,
) -> str:
    """
    Gửi transcript ASR bị lỗi đến RAG với instruction sửa lỗi rõ ràng.
    RAG sẽ retrieve chunks liên quan và dùng LLM để correction thay vì chỉ summarize.

    Args:
        text_window: Transcript ghép từ k segment trước làm câu query.
        subject:     Môn học để filter đúng subject trong RAG.
        rag_api_url: URL của RAG API service.
        timeout:     Timeout tính bằng giây (mặc định 20s).

    Returns:
        Chuỗi context đã được correction (~400 ký tự), hoặc rỗng nếu lỗi/timeout.
    """
    if not rag_api_url or not text_window.strip():
        return ""

    # Gửi với instruction correction rõ ràng thay vì raw transcript
    # Giúp LLM trong RAG hiểu nhiệm vụ là sửa lỗi, không phải tìm kiếm
    domain_label = f"môn {subject}" if subject else "học thuật"
    correction_query = (
        f"Đây là transcript ASR tiếng Việt {domain_label} bị lỗi chính tả và thuật ngữ. "
        f"Hãy sửa lại các thuật ngữ chuyên ngành cho đúng:\n\n{text_window[:300]}"
    )

    try:
        resp = requests.post(
            f"{rag_api_url.rstrip('/')}/query",
            json={
                "question": correction_query,
                "subject": subject,
                "model_key": "qwen3-8b",
                "use_history": False,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            answer = resp.json().get("answer", "")
            # Strip phần nguồn tham khảo RAG tự thêm vào cuối
            answer = re.sub(r'\n\n📚.*$', '', answer, flags=re.DOTALL).strip()
            # Cắt bớt để không làm phình system prompt quá lớn
            return answer[:400].strip()
        else:
            print(f"⚠️ RAG API returned status {resp.status_code}, skipping context")
    except requests.exceptions.Timeout:
        print(f"⚠️ RAG context query timed out after {timeout}s, continuing without context")
    except requests.exceptions.ConnectionError:
        print(f"⚠️ RAG API unreachable at {rag_api_url}, continuing without context")
    except Exception as e:
        print(f"⚠️ RAG context query failed: {e}")

    return ""


# =============================================================================
# LLM Refinement
# =============================================================================

def refine_transcription_batch(
    segments: list,
    language: str = "vi",
    llm_proxy_url: str = "http://localhost:5000",
    llm_model: str = "Qwen/Qwen3-8B-AWQ",
    llm_timeout_seconds: int = 60,
    batch_size: int = 20,
    custom_prompt_template: str = None,
    subject: str = None,
    rag_api_url: str = "",
    rag_context_window: int = 5,
) -> list:
    """
    Refine transcription segments bằng LLM (Qwen3 8B) với RAG context injection.

    Với mỗi batch:
      1. Lấy transcript của `rag_context_window` segment TRƯỚC batch hiện tại làm query.
      2. Query RAG API với instruction correction rõ ràng để lấy thuật ngữ đúng.
      3. Inject context đó vào system prompt của Qwen3 để sửa đúng thuật ngữ.

    Batch đầu tiên (i=0) sẽ không có prev_text → RAG trả rỗng → Qwen3 vẫn chạy
    bình thường với default prompt, không ảnh hưởng pipeline.

    Fallback về segment gốc nếu LLM lỗi hoặc RAG timeout.
    """
    if not segments:
        return []

    refined_segments = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]

        # ── CB-RAG style: dùng transcript các segment TRƯỚC làm query ──────────
        prev_segments = segments[max(0, i - rag_context_window):i]
        prev_text = " ".join(s.text.strip() for s in prev_segments)

        rag_context = ""
        if rag_api_url and prev_text:
            print(f"🔍 Querying RAG for batch {i // batch_size + 1} context (subject: {subject})...")
            rag_context = query_rag_context(
                prev_text,
                subject=subject,
                rag_api_url=rag_api_url,
                timeout=20,
            )
            if rag_context:
                print(f"✅ RAG context retrieved ({len(rag_context)} chars) for batch {i // batch_size + 1}")
            else:
                print(f"ℹ️ No RAG context for batch {i // batch_size + 1}, using default prompt")

        # ── Build system prompt, inject RAG context nếu có ───────────────────
        system_prompt = custom_prompt_template or get_default_prompt_template(language)
        if rag_context:
            system_prompt = (
                system_prompt
                + "\n\nNGỮ CẢNH TÀI LIỆU (thuật ngữ và khái niệm đúng từ slide bài giảng):\n"
                + rag_context
                + "\nƯu tiên dùng đúng các thuật ngữ này khi sửa chính tả transcript."
            )

        # ── Prepare batch JSON ─────────────────────────────────────────────────
        batch_data = [
            {
                "id": i + j,
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
            }
            for j, segment in enumerate(batch)
        ]

        batch_json_str = json.dumps(batch_data, ensure_ascii=False)
        user_message = f"Refine these transcription segments:\n\n{batch_json_str}"

        try:
            assistant_message = None
            for endpoint in _resolve_llm_urls(llm_proxy_url):
                response = requests.post(
                    endpoint,
                    json={
                        "model": llm_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2500,
                        "top_p": 0.9,
                    },
                    timeout=llm_timeout_seconds,
                )

                if response.status_code != 200:
                    continue

                payload = response.json()
                if "choices" in payload:
                    assistant_message = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
                elif "response" in payload:
                    assistant_message = payload.get("response", "")
                elif "text" in payload:
                    assistant_message = payload.get("text", "")

                if assistant_message:
                    break

            if not assistant_message:
                print("⚠️ LLM proxy unavailable or response format unsupported, fallback to original")
                refined_segments.extend(batch)
                continue

            refined_data = _extract_json_array(assistant_message)
            if not isinstance(refined_data, list):
                print("⚠️ LLM response does not contain valid JSON array, fallback to original")
                refined_segments.extend(batch)
                continue

            refined_map = {}
            for item in refined_data:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id")
                item_text = str(item.get("text", "")).strip()
                if isinstance(item_id, int) and item_text:
                    refined_map[item_id] = item_text

            for j, seg in enumerate(batch):
                seg_id = i + j
                refined_text = refined_map.get(seg_id)
                if refined_text and 3 <= len(refined_text) <= 300:
                    seg.text = clean_repetitive_text(refined_text)

            refined_segments.extend(batch)

        except (requests.RequestException, requests.Timeout) as e:
            print(f"⚠️ LLM refinement timeout/error: {e}, fallback to original")
            refined_segments.extend(batch)

    return refined_segments


def smart_segment_split(seg, min_chars=50, max_chars=70, min_duration=1.0):
    """Chia segment theo target 50-70 ký tự, ưu tiên ngắt ở dấu câu"""
    text = clean_repetitive_text(seg.text.strip())

    if len(text) <= max_chars:
        return [(seg.start, seg.end, text)]

    words = text.split()
    parts = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if len(candidate) > max_chars and current:
            parts.append(current.strip())
            current = word
            continue

        current = candidate

        if len(current) >= min_chars and re.search(r"[.!?;:,]$", current):
            parts.append(current.strip())
            current = ""

    if current:
        parts.append(current.strip())

    normalized_parts = []
    for part in parts:
        if not normalized_parts:
            normalized_parts.append(part)
            continue

        if len(part) < min_chars:
            merged = f"{normalized_parts[-1]} {part}".strip()
            if len(merged) <= max_chars + 20:
                normalized_parts[-1] = merged
            else:
                normalized_parts.append(part)
        else:
            normalized_parts.append(part)

    parts = normalized_parts

    if not parts:
        return [(seg.start, seg.end, text)]

    total_duration = seg.end - seg.start
    duration_per_part = max(total_duration / len(parts), min_duration)

    result = []
    for i, part in enumerate(parts):
        part_start = seg.start + i * duration_per_part
        part_end = min(seg.start + (i + 1) * duration_per_part, seg.end)
        if part.strip():
            result.append((part_start, part_end, part.strip()))

    return result


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except FileNotFoundError:
        return False


def is_m3u8_url(url: str) -> bool:
    return url.lower().endswith('.m3u8') or 'm3u8' in url.lower()


def download_m3u8_stream(url: str) -> str:
    print(f"⬇️ Processing M3U8 stream from {url}...")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.close()

    try:
        cmd = [
            "ffmpeg",
            "-i", url,
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            "-avoid_negative_ts", "make_zero",
            "-y",
            temp_file.name
        ]

        print(f"🔧 Running ffmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode != 0:
            raise Exception(f"FFmpeg failed with error: {result.stderr}")

        if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
            raise Exception("Downloaded file is empty or doesn't exist")

        print(f"✅ M3U8 stream downloaded successfully: {temp_file.name}")
        return temp_file.name

    except subprocess.TimeoutExpired:
        raise Exception("Download timeout - stream took too long to process")
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise Exception(f"Failed to download M3U8 stream: {str(e)}")


def download_temp_file(url: str) -> str:
    if is_google_drive_url(url):
        url_type, resource_id = parse_google_drive_url(url)
        if url_type == "folder":
            raise Exception("Folder link is not supported in /convert. Use /convert-drive-folder")
        print(f"⬇️ Downloading Google Drive file {resource_id}...")
        return download_google_drive_file(resource_id)

    if is_m3u8_url(url):
        return download_m3u8_stream(url)
    else:
        print(f"⬇️ Downloading media from {url}...")
        last_error = None

        for attempt in range(1, _download_max_retries + 1):
            temp_file_path = None
            response = None
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) whisper-api/1.0",
                    "Connection": "close",
                }
                response = requests.get(
                    url,
                    stream=True,
                    timeout=(_direct_download_connect_timeout, _direct_download_read_timeout),
                    headers=headers,
                )
                response.raise_for_status()

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file_path = temp_file.name
                temp_file.close()

                with open(temp_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                    raise Exception("Downloaded media file is empty")

                return temp_file_path

            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.RequestException,
            ) as e:
                last_error = e
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                if attempt < _download_max_retries:
                    print(
                        f"⚠️ Direct download attempt {attempt}/{_download_max_retries} failed: {e}. Retrying..."
                    )
                    _sleep_before_retry(attempt)
                    continue
                break
            except Exception as e:
                last_error = e
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                if attempt < _download_max_retries:
                    print(
                        f"⚠️ Direct download attempt {attempt}/{_download_max_retries} failed: {e}. Retrying..."
                    )
                    _sleep_before_retry(attempt)
                    continue
                break
            finally:
                try:
                    if response is not None:
                        response.close()
                except Exception:
                    pass

        raise Exception(
            f"Direct media download failed after {_download_max_retries} attempts: {last_error}"
        )


def get_filename_from_url(url: str) -> str:
    if is_google_drive_url(url):
        try:
            url_type, resource_id = parse_google_drive_url(url)
            if url_type == "file":
                drive_name = _drive_file_name_cache.get(resource_id)
                if not drive_name:
                    service = get_drive_service()
                    metadata = service.files().get(
                        fileId=resource_id,
                        fields="name",
                        supportsAllDrives=True,
                    ).execute(num_retries=_google_api_retries)
                    drive_name = metadata.get("name", f"drive_{resource_id}")
                base_name = os.path.splitext(drive_name)[0]
                return sanitize_filename(base_name)
        except Exception:
            pass

    parsed = urlparse(url)
    base_name = os.path.splitext(os.path.basename(parsed.path))[0]

    if is_m3u8_url(url) and not base_name:
        domain = parsed.netloc.replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"m3u8_{domain}_{timestamp}"

    return sanitize_filename(base_name or f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def build_initial_prompt(language: str, subject: Optional[str] = None) -> Optional[str]:
    """
    Build initial_prompt cho Whisper để bias nhận dạng đúng ngữ cảnh bài giảng.
    Subject được dùng để gợi ý domain, không cần liệt kê từng thuật ngữ thủ công.
    """
    if language != "vi":
        if subject:
            return f"Academic lecture on {subject}. Technical terminology expected."
        return None

    base = "Đây là bài giảng học thuật tiếng Việt"
    if subject:
        base += f" môn {subject}"
    return base + ". Nội dung mang tính học thuật, chuyên ngành."


def process_video_transcription(job_id: str, video_url: str, language: str, request: TranscriptionOptions):
    """Enhanced background task với anti-hallucination và RAG context injection"""
    input_file = None
    try:
        update_job_status(job_id, status="processing")

        if is_m3u8_url(video_url):
            update_job_status(job_id, progress="Processing M3U8 stream...")
        else:
            update_job_status(job_id, progress="Downloading video...")

        input_file = download_temp_file(video_url)
        base_name = get_filename_from_url(video_url)

        output_dir = os.path.join("file_vtt", base_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}.vtt")
        txt_output_file = os.path.join(output_dir, f"{base_name}.txt")

        update_job_status(job_id, progress="Loading Whisper model...")

        model = _get_thread_model(request.model)

        vad_parameters = None
        if request.enable_vad:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400
            }

        update_job_status(job_id, progress="Transcribing audio with anti-hallucination...")

        segments, info = model.transcribe(
            input_file,
            language=language,
            task="transcribe",
            beam_size=request.beam_size,
            patience=request.patience,
            length_penalty=request.length_penalty,
            repetition_penalty=request.repetition_penalty,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
            temperature=request.temperature,
            compression_ratio_threshold=request.compression_ratio_threshold,
            log_prob_threshold=request.log_prob_threshold,
            no_speech_threshold=request.no_speech_threshold,
            condition_on_previous_text=request.condition_on_previous_text,
            initial_prompt=build_initial_prompt(language, request.subject),
            vad_filter=request.enable_vad,
            vad_parameters=vad_parameters,
            word_timestamps=True,
        )

        update_job_status(job_id, progress="Processing and filtering segments...")

        valid_segments = []
        hallucination_count = 0
        total_segments_count = 0

        for segment in segments:
            total_segments_count += 1
            if validate_segment_quality(segment):
                cleaned_text = clean_repetitive_text(segment.text)
                if cleaned_text.strip():
                    segment.text = cleaned_text
                    valid_segments.append(segment)
            else:
                hallucination_count += 1
                print(f"⚠️ Filtered hallucination: {segment.text[:50]}...")

        # === LLM Refinement + RAG Context Injection ===
        if request.enable_llm_refine and valid_segments:
            rag_enabled = bool(request.rag_api_url)
            progress_msg = "Refining transcript with LLM + RAG context..." if rag_enabled else "Refining transcript with LLM (Qwen3)..."
            update_job_status(job_id, progress=progress_msg)

            if rag_enabled:
                print(f"🔗 RAG integration enabled — API: {request.rag_api_url}, subject: {request.subject}")

            try:
                valid_segments = refine_transcription_batch(
                    valid_segments,
                    language=request.language,
                    llm_proxy_url=request.llm_proxy_url,
                    llm_model=request.llm_model,
                    llm_timeout_seconds=request.llm_timeout_seconds,
                    batch_size=request.refine_batch_size,
                    custom_prompt_template=request.prompt_template,
                    subject=request.subject,
                    rag_api_url=request.rag_api_url or "",
                    rag_context_window=request.rag_context_window,
                )
                print(f"✅ LLM refinement completed for {len(valid_segments)} segments")
            except Exception as e:
                print(f"⚠️ LLM refinement failed, continuing with unrefined segments: {e}")

        update_job_status(job_id, progress="Generating optimized VTT and TXT files...")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in valid_segments:
                sub_segments = smart_segment_split(segment, min_chars=50, max_chars=70)
                for sub_start, sub_end, sub_text in sub_segments:
                    if sub_text.strip():
                        f.write(f"{format_timestamp(sub_start, vtt=True)} --> {format_timestamp(sub_end, vtt=True)}\n")
                        f.write(f"{sub_text}\n\n")

        with open(txt_output_file, "w", encoding="utf-8") as f:
            transcript_parts = []
            for segment in valid_segments:
                seg_text = segment.text.strip()
                if seg_text:
                    if request.txt_include_timestamps:
                        timestamp = format_timestamp(segment.start)
                        transcript_parts.append(f"[{timestamp}] {seg_text}")
                    else:
                        transcript_parts.append(seg_text)

            if request.txt_include_timestamps:
                f.write("\n".join(transcript_parts))
            else:
                f.write(" ".join(transcript_parts))

        result_payload = {
            "vtt_filename": f"{base_name}.vtt",
            "txt_filename": f"{base_name}.txt",
            "vtt_download_url": f"/download/{base_name}.vtt",
            "txt_download_url": f"/download/{base_name}.txt",
            "vtt_file_path": output_file,
            "txt_file_path": txt_output_file,
            "created_at": datetime.now().isoformat(),
            "source_type": "m3u8_stream" if is_m3u8_url(video_url) else "direct_file",
            "llm_refined": request.enable_llm_refine,
            "rag_context_used": bool(request.rag_api_url and request.enable_llm_refine),
            "rag_subject": request.subject,
            "llm_prompt_template": "custom" if request.prompt_template else "default",
            "stats": {
                "total_segments": total_segments_count,
                "valid_segments": len(valid_segments),
                "filtered_hallucinations": hallucination_count,
                "detected_language": info.language if hasattr(info, 'language') else request.language,
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else None,
            },
        }
        update_job_status(
            job_id,
            status="completed",
            progress="Completed successfully",
            error=None,
            result=result_payload,
        )
        remove_job_payload(job_id)

        rag_note = f" — RAG context: {request.rag_api_url}" if request.rag_api_url else ""
        print(
            f"✅ Job {job_id} completed"
            f" — VTT + TXT generated"
            f" — Filtered {hallucination_count} hallucinations"
            + (f" — LLM refined" if request.enable_llm_refine else "")
            + rag_note
        )

    except Exception as e:
        update_job_status(job_id, status="failed", error=str(e))
        remove_job_payload(job_id)
        print(f"❌ Job {job_id} failed: {e}")
    finally:
        if input_file and os.path.exists(input_file):
            os.remove(input_file)