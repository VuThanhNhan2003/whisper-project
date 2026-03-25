import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, HttpUrl


class TranscriptionOptions(BaseModel):
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
    no_speech_threshold: float = 0.6S
    
    # === LLM Refinement Options (for .txt quality + RAG) ===
    enable_llm_refine: bool = True
    llm_proxy_url: str = Field(
        default_factory=lambda: os.getenv(
            "LLM_PROXY_URL", "http://host.docker.internal:5000"))
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_seconds: int = 60
    refine_batch_size: int = 20
    prompt_template: Optional[str] = None  # Custom prompt for mixed lang/code/LaTeX handling
    txt_include_timestamps: bool = False


class VideoConvertRequest(TranscriptionOptions):
    video_url: HttpUrl


class DriveFolderConvertRequest(TranscriptionOptions):
    folder_url: HttpUrl


class ConvertResponse(BaseModel):
    job_id: str
    message: str
    status: str


class BatchConvertResponse(BaseModel):
    job_ids: list[str]
    total_jobs: int
    message: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
