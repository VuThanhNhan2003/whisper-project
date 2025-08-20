import os
import uuid
import tempfile
import subprocess
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from faster_whisper import WhisperModel, BatchedInferencePipeline

# --- Models ---
class VideoConvertRequest(BaseModel):
    video_url: HttpUrl
    language: str = "vi"
    model: str = "large-v3-turbo"
    # Thêm các tùy chọn chống hallucination
    enable_vad: bool = True
    condition_on_previous_text: bool = False
    beam_size: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    temperature: float = 0.0  # Giảm randomness
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

class ConvertResponse(BaseModel):
    job_id: str
    message: str
    status: str

class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

# --- FastAPI App ---
app = FastAPI(
    title="Advanced Whisper Video Transcription API",
    description="API để chuyển đổi video thành phụ đề VTT và TXT với khả năng chống hallucination",
    version="2.0.0"
)

# --- Global Variables ---
jobs_status: Dict[str, JobStatus] = {}
executor = ThreadPoolExecutor(max_workers=2)

# --- Anti-Hallucination Helpers ---
def detect_hallucination_patterns(text: str) -> bool:
    """Phát hiện các pattern hallucination phổ biến"""
    # Pattern 1: Lặp ký tự hoặc số quá nhiều
    if re.search(r'(.)\1{10,}', text):  # Ký tự lặp > 10 lần
        return True
    
    # Pattern 2: Lặp từ quá nhiều
    words = text.split()
    if len(words) > 5:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > len(words) * 0.3:  # Từ chiếm >30% câu
                return True
    
    # Pattern 3: Text quá ngắn cho segment dài (>30s)
    if len(text.strip()) < 10:
        return True
    
    # Pattern 4: Các cụm từ promotional phổ biến (Vietnamese)
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
    # Loại bỏ ký tự lặp
    text = re.sub(r'(.)\1{5,}', r'\1', text)
    
    # Loại bỏ từ lặp liên tiếp
    words = text.split()
    cleaned_words = []
    prev_word = ""
    repeat_count = 0
    
    for word in words:
        if word.lower() == prev_word.lower():
            repeat_count += 1
            if repeat_count < 2:  # Cho phép lặp tối đa 1 lần
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            repeat_count = 0
        prev_word = word
    
    return ' '.join(cleaned_words)

def validate_segment_quality(segment, min_duration=0.5, max_duration=30.0) -> bool:
    """Kiểm tra chất lượng segment"""
    duration = segment.end - segment.start
    
    # Segment quá ngắn hoặc quá dài
    if duration < min_duration or duration > max_duration:
        return False
    
    # Text quá ngắn cho segment dài
    if duration > 10 and len(segment.text.strip()) < 20:
        return False
    
    # Kiểm tra hallucination
    if detect_hallucination_patterns(segment.text):
        return False
    
    return True

# --- Enhanced Helpers ---
def format_timestamp(seconds, vtt=False):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    sep = '.' if vtt else ','
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millisecs:03d}"

def smart_segment_split(seg, max_chars=60, min_duration=1.0):
    """Chia segment thông minh hơn, tránh cắt giữa từ"""
    text = clean_repetitive_text(seg.text.strip())
    
    if len(text) <= max_chars:
        return [(seg.start, seg.end, text)]

    # Chia theo câu trước, sau đó mới chia theo từ
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 1:
        parts = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current + " " + sentence) <= max_chars:
                current = current + ". " + sentence if current else sentence
            else:
                if current:
                    parts.append(current.strip())
                current = sentence
        if current:
            parts.append(current.strip())
    else:
        # Chia theo từ nhưng tránh cắt giữa cụm từ
        words = text.split()
        parts = []
        current = ""
        for word in words:
            if len(current + " " + word) <= max_chars:
                current = current + " " + word if current else word
            else:
                if current:
                    parts.append(current.strip())
                current = word
        if current:
            parts.append(current.strip())

    if not parts:
        return [(seg.start, seg.end, text)]

    total_duration = seg.end - seg.start
    duration_per_part = max(total_duration / len(parts), min_duration)

    result = []
    for i, part in enumerate(parts):
        part_start = seg.start + i * duration_per_part
        part_end = min(seg.start + (i + 1) * duration_per_part, seg.end)
        if part.strip():  # Chỉ thêm phần có nội dung
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
            "-avoid_negative_ts", "make_zero",  # Tránh timestamp âm
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
    if is_m3u8_url(url):
        return download_m3u8_stream(url)
    else:
        print(f"⬇️ Downloading media from {url}...")
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code != 200:
            raise Exception(f"Download failed with status code {r.status_code}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_file.name

def get_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    base_name = os.path.splitext(os.path.basename(parsed.path))[0]
    
    if is_m3u8_url(url) and not base_name:
        domain = parsed.netloc.replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"m3u8_{domain}_{timestamp}"
    
    return base_name or f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def process_video_transcription(job_id: str, video_url: str, language: str, request: VideoConvertRequest):
    """Enhanced background task với anti-hallucination"""
    try:
        jobs_status[job_id].status = "processing"
        
        if is_m3u8_url(video_url):
            jobs_status[job_id].progress = "Processing M3U8 stream..."
        else:
            jobs_status[job_id].progress = "Downloading video..."
        
        input_file = download_temp_file(video_url)
        base_name = get_filename_from_url(video_url)
        
        output_dir = os.path.join("File vtt", base_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}.vtt")
        # Thêm file TXT output
        txt_output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        jobs_status[job_id].progress = "Loading Whisper model..."
        
        # Load model với cấu hình tối ưu cho CPU
        model = WhisperModel(
            request.model, 
            device="cpu",  # Force CPU vì server không có GPU
            compute_type="int8",  # Tối ưu cho CPU
            num_workers=4  # Sử dụng 4 cores
        )
        
        # Cấu hình VAD nếu được bật
        vad_parameters = None
        if request.enable_vad:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400
            }
        
        jobs_status[job_id].progress = "Transcribing audio with anti-hallucination..."
        
        # Transcribe với các tham số chống hallucination
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
            initial_prompt="Đây là một video học thuật tiếng Việt về khoa học." if language == "vi" else None,
            vad_filter=request.enable_vad,
            vad_parameters=vad_parameters,
            word_timestamps=True  # Enable để có thể phân tích tốt hơn
        )
        
        jobs_status[job_id].progress = "Processing and filtering segments..."
        
        # Filter và clean segments
        valid_segments = []
        hallucination_count = 0
        
        for segment in segments:
            if validate_segment_quality(segment):
                # Clean text trước khi thêm
                cleaned_text = clean_repetitive_text(segment.text)
                if cleaned_text.strip():
                    segment.text = cleaned_text
                    valid_segments.append(segment)
            else:
                hallucination_count += 1
                print(f"⚠️ Filtered hallucination: {segment.text[:50]}...")
        
        jobs_status[job_id].progress = "Generating optimized VTT and TXT files..."
        
        # Generate VTT với smart splitting
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in valid_segments:
                sub_segments = smart_segment_split(segment, max_chars=60)
                for sub_start, sub_end, sub_text in sub_segments:
                    if sub_text.strip():  # Chỉ ghi segment có nội dung
                        f.write(f"{format_timestamp(sub_start, vtt=True)} --> {format_timestamp(sub_end, vtt=True)}\n")
                        f.write(f"{sub_text}\n\n")
        
        # Generate TXT file với transcript sạch (cho RAG pipeline)
        with open(txt_output_file, "w", encoding="utf-8") as f:
            # Chỉ ghi transcript thuần túy, không có metadata
            transcript_parts = []
            for segment in valid_segments:
                cleaned_text = clean_repetitive_text(segment.text).strip()
                if cleaned_text:
                    transcript_parts.append(cleaned_text)
            
            # Ghi toàn bộ transcript thành một đoạn văn liên tục
            f.write(" ".join(transcript_parts))
        
        # Cleanup
        if os.path.exists(input_file):
            os.remove(input_file)
        
        # Final status với thống kê - cập nhật để bao gồm file TXT
        jobs_status[job_id].status = "completed"
        jobs_status[job_id].progress = "Completed successfully"
        jobs_status[job_id].result = {
            "vtt_filename": f"{base_name}.vtt",
            "txt_filename": f"{base_name}.txt",
            "vtt_download_url": f"/download/{base_name}.vtt",
            "txt_download_url": f"/download/{base_name}.txt",
            "vtt_file_path": output_file,
            "txt_file_path": txt_output_file,
            "created_at": datetime.now().isoformat(),
            "source_type": "m3u8_stream" if is_m3u8_url(video_url) else "direct_file",
            "stats": {
                "total_segments": len(list(segments)) if segments else 0,
                "valid_segments": len(valid_segments),
                "filtered_hallucinations": hallucination_count,
                "detected_language": info.language if hasattr(info, 'language') else language,
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else None
            }
        }
        
        print(f"✅ Job {job_id} completed - Generated VTT and TXT files - Filtered {hallucination_count} hallucinations")
        
    except Exception as e:
        jobs_status[job_id].status = "failed"
        jobs_status[job_id].error = str(e)
        print(f"❌ Job {job_id} failed: {e}")

# --- Enhanced Endpoints ---
@app.get("/health")
async def health_check():
    ffmpeg_available = check_ffmpeg()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ffmpeg_available": ffmpeg_available,
        "active_jobs": len([j for j in jobs_status.values() if j.status == "processing"]),
        "supported_formats": ["MP4", "AVI", "MOV", "M3U8", "HLS"],
        "output_formats": ["VTT", "TXT"],
        "anti_hallucination": "enabled",
        "server_type": "CPU_optimized"
    }

@app.post("/convert", response_model=ConvertResponse)
async def convert_video(request: VideoConvertRequest, background_tasks: BackgroundTasks):
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg not available on server")
    
    job_id = str(uuid.uuid4())
    
    jobs_status[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress="Job queued for processing with anti-hallucination enabled"
    )
    
    background_tasks.add_task(
        process_video_transcription,
        job_id,
        str(request.video_url),
        request.language,
        request  # Pass full request object
    )
    
    return ConvertResponse(
        job_id=job_id,
        message="Video conversion started with enhanced quality filtering - will generate both VTT and TXT files",
        status="queued"
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_status[job_id]

@app.get("/list")
async def list_files():
    vtt_dir = "File vtt"
    if not os.path.exists(vtt_dir):
        return {"files": []}
    
    files = []
    for folder_name in os.listdir(vtt_dir):
        folder_path = os.path.join(vtt_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(('.vtt', '.txt')):
                    file_path = os.path.join(folder_path, file_name)
                    stat = os.stat(file_path)
                    file_type = "VTT" if file_name.endswith('.vtt') else "TXT"
                    files.append({
                        "filename": file_name,
                        "folder": folder_name,
                        "type": file_type,
                        "download_url": f"/download/{file_name}",
                        "size": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
    
    return {"files": files}

@app.get("/download/{filename}")
async def download_file(filename: str):
    vtt_dir = "File vtt"
    file_path = None
    
    for folder_name in os.listdir(vtt_dir):
        folder_path = os.path.join(vtt_dir, folder_name)
        if os.path.isdir(folder_path):
            potential_path = os.path.join(folder_path, filename)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Xác định media type dựa trên extension
    if filename.endswith('.vtt'):
        media_type = "text/vtt"
    elif filename.endswith('.txt'):
        media_type = "text/plain"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_status[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/")
async def root():
    return {
        "message": "Advanced Whisper Video Transcription API v2.0",
        "features": [
            "Anti-hallucination filtering",
            "Smart text segmentation", 
            "VAD preprocessing",
            "CPU optimized processing",
            "M3U8 stream support",
            "Dual output: VTT + TXT files"
        ],
        "endpoints": {
            "POST /convert": "Convert video to VTT subtitle and TXT transcript with quality filtering",
            "GET /status/{job_id}": "Check job status with detailed stats",
            "GET /list": "List all VTT and TXT files",
            "GET /download/{filename}": "Download VTT or TXT file",
            "GET /health": "Health check"
        },
        "output_formats": {
            "VTT": "WebVTT subtitle file with timestamps - suitable for video players",
            "TXT": "Clean plain text transcript - optimized for RAG pipeline and NLP processing"
        },
        "example": {
            "basic_request": {
                "video_url": "https://example.com/video.mp4",
                "language": "vi"
            },
            "advanced_request": {
                "video_url": "https://example.com/video.mp4",
                "language": "vi",
                "model": "large-v3-turbo",
                "enable_vad": True,
                "temperature": 0.0,
                "beam_size": 5,
                "no_speech_threshold": 0.6
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
