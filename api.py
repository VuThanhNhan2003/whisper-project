import os
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from pipeline import (
    check_ffmpeg,
    create_job_status,
    enqueue_transcription_job,
    get_processing_metrics,
    initialize_job_store,
    is_google_drive_url,
    jobs_status,
    list_google_drive_video_files,
    parse_google_drive_url,
    remove_job_status,
    start_job_workers,
    update_job_status,
)
from schemas import (
    BatchConvertResponse,
    ConvertResponse,
    DriveFolderConvertRequest,
    JobStatus,
    VideoConvertRequest,
)


app = FastAPI(
    title="Advanced Whisper Video Transcription API",
    description="API để chuyển đổi video thành phụ đề VTT và TXT với khả năng chống hallucination",
    version="2.0.0"
)


@app.get("/health")
async def health_check():
    ffmpeg_available = check_ffmpeg()
    metrics = get_processing_metrics()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ffmpeg_available": ffmpeg_available,
        "active_jobs": metrics["processing_jobs"],
        "queued_jobs": metrics["queued_jobs"],
        "worker_count": metrics["worker_count"],
        "queue_size": metrics["queue_size"],
        "queue_capacity": metrics["queue_capacity"],
        "supported_formats": ["MP4", "AVI", "MOV", "M3U8", "HLS"],
        "output_formats": ["VTT", "TXT"],
        "anti_hallucination": "enabled",
        "server_type": "CPU_optimized"
    }


@app.on_event("startup")
async def startup_event():
    initialize_job_store()
    start_job_workers()


@app.post("/convert", response_model=ConvertResponse)
async def convert_video(request: VideoConvertRequest):
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg not available on server")

    if is_google_drive_url(str(request.video_url)):
        try:
            url_type, _ = parse_google_drive_url(str(request.video_url))
            if url_type == "folder":
                raise HTTPException(
                    status_code=400,
                    detail="Drive folder URL is not supported in /convert. Use /convert-drive-folder",
                )
        except HTTPException:
            raise
        except Exception:
            # Keep backward behavior for non-standard links and let pipeline handle detailed errors.
            pass

    job_id = str(uuid.uuid4())

    create_job_status(job_id, JobStatus(
        job_id=job_id,
        status="queued",
        progress="Job queued for processing with anti-hallucination enabled"
    ))

    try:
        enqueue_transcription_job(
            job_id,
            str(request.video_url),
            request.language,
            request,
        )
    except Exception as e:
        update_job_status(job_id, status="failed", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))

    return ConvertResponse(
        job_id=job_id,
        message="Video conversion started with enhanced quality filtering - will generate both VTT and TXT files",
        status="queued"
    )


@app.post("/convert-drive-folder", response_model=BatchConvertResponse)
async def convert_drive_folder(request: DriveFolderConvertRequest):
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg not available on server")

    try:
        drive_files = list_google_drive_video_files(str(request.folder_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Google Drive folder: {e}")

    if not drive_files:
        raise HTTPException(status_code=404, detail="No video files found in the provided Drive folder")

    job_ids = []
    skipped = 0
    for file_info in drive_files:
        job_id = str(uuid.uuid4())
        create_job_status(job_id, JobStatus(
            job_id=job_id,
            status="queued",
            progress=f"Queued from Drive folder: {file_info['name']}",
        ))

        try:
            enqueue_transcription_job(
                job_id,
                f"gdrive://file/{file_info['id']}",
                request.language,
                request,
            )
            job_ids.append(job_id)
        except Exception as e:
            skipped += 1
            update_job_status(job_id, status="failed", error=str(e), progress="Not queued")

    return BatchConvertResponse(
        job_ids=job_ids,
        total_jobs=len(job_ids),
        message=(
            f"Drive folder queued successfully: {len(job_ids)} job(s) queued"
            + (f", {skipped} skipped due to queue capacity" if skipped else "")
        ),
        status="queued",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_status[job_id]


@app.get("/list")
async def list_files():
    vtt_dir = "file_vtt"
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
    vtt_dir = "file_vtt"
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
    if not remove_job_status(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
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
            "Google Drive file and folder support",
            "Dual output: VTT + TXT files"
        ],
        "endpoints": {
            "POST /convert": "Convert video to VTT subtitle and TXT transcript with quality filtering",
            "POST /convert-drive-folder": "Queue all video files in a Google Drive folder",
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
            "drive_folder_request": {
                "folder_url": "https://drive.google.com/drive/folders/1g-ZDze6jVI_Y418_O7vHh7QPqI0JXOjq",
                "language": "vi"
            },
            "advanced_request": {
                "video_url": "https://example.com/video.mp4",
                "language": "vi",
                "model": "large-v3-turbo",
                "enable_llm_refine": True,
                "llm_proxy_url": "http://host.docker.internal:5000",
                "llm_model": "Qwen/Qwen3-8B-AWQ",
                "txt_include_timestamps": False,
                "enable_vad": True,
                "temperature": 0.0,
                "beam_size": 5,
                "no_speech_threshold": 0.6
            }
        }
    }
