import os
import json
import requests
import tempfile
import subprocess
from urllib.parse import urlparse
from faster_whisper import WhisperModel, BatchedInferencePipeline

# --- Helpers ---
def format_timestamp(seconds, vtt=False):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    sep = '.' if vtt else ','
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millisecs:03d}"

def split_segment_by_chars(seg, max_chars=80):
    text = seg.text.strip()
    if len(text) <= max_chars:
        return [(seg.start, seg.end, text)]

    words = text.split()
    parts = []
    current = ""
    for word in words:
        if len(current + " " + word) <= max_chars:
            current = current + " " + word if current else word
        else:
            parts.append(current.strip())
            current = word
    if current:
        parts.append(current.strip())

    total_duration = seg.end - seg.start
    duration_per_part = total_duration / len(parts)

    result = []
    for i, part in enumerate(parts):
        part_start = seg.start + i * duration_per_part
        part_end = seg.start + (i + 1) * duration_per_part
        result.append((part_start, part_end, part))
    return result

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except FileNotFoundError:
        print("⚠️ FFmpeg not found! Please install it.")
        return False

def download_temp_file(url):
    print(f"⬇️  Downloading media from {url}...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Download failed with status code {r.status_code}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_file.name, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_file.name

def get_filename_from_url(url):
    parsed = urlparse(url)
    return os.path.splitext(os.path.basename(parsed.path))[0]

# --- Load config ---
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

input_url = config["input_url"]
output_format = config.get("output_format", "srt")
language = config.get("language", "vi")
model_size = config.get("model", "large-v3")
device = config.get("device", "auto")
task = config.get("task", "transcribe")
initial_prompt = config.get("initial_prompt", "")
max_chars = config.get("subtitle_options", {}).get("max_chars", 80)

# --- Check ---
if not check_ffmpeg():
    exit(1)

# --- Download file ---
input_file = download_temp_file(input_url)

# --- Determine output filename ---
base_name = get_filename_from_url(input_url)

# Tạo thư mục lưu file .vtt theo cấu trúc File vtt/<ten-file>/
if output_format == "vtt":
    output_dir = os.path.join("File vtt", base_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{base_name}.{output_format}")
else:
    output_file = f"{base_name}.{output_format}"

# --- Load model ---
print(f"🚀 Loading faster-whisper model: {model_size} on {device}...")
model = WhisperModel(model_size, device=device, compute_type="int8")
pipeline = BatchedInferencePipeline(model=model)

print(f"🎧 Transcribing audio...")
segments, _ = pipeline.transcribe(
    input_file,
    language=language,
    task=task,
    beam_size=5,
    initial_prompt=initial_prompt if initial_prompt else None
)

# --- Save subtitle ---
with open(output_file, "w", encoding="utf-8") as f:
    if output_format == "srt":
        idx = 1
        for seg in segments:
            sub_segments = split_segment_by_chars(seg, max_chars=max_chars)
            for sub_start, sub_end, sub_text in sub_segments:
                f.write(f"{idx}\n")
                f.write(f"{format_timestamp(sub_start)} --> {format_timestamp(sub_end)}\n")
                f.write(f"{sub_text}\n\n")
                idx += 1
    elif output_format == "vtt":
        f.write("WEBVTT\n\n")
        for seg in segments:
            sub_segments = split_segment_by_chars(seg, max_chars=max_chars)
            for sub_start, sub_end, sub_text in sub_segments:
                f.write(f"{format_timestamp(sub_start, vtt=True)} --> {format_timestamp(sub_end, vtt=True)}\n")
                f.write(f"{sub_text}\n\n")
    else:
        for seg in segments:
            f.write(f"[{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}] {seg.text.strip()}\n")

print(f"✅ Subtitle saved to {os.path.abspath(output_file)}")

# --- Cleanup ---
if os.path.exists(input_file):
    os.remove(input_file)
    print("🧹 Temporary file cleaned up.")
