import datetime
import json
import os
import re
import subprocess
from typing import BinaryIO, Dict

from faster_whisper import WhisperModel
from numpy import ndarray


class ModelConfig():
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


# Logic
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


# def download_m3u8_stream(url: str) -> str:
#     print(f"⬇️ Processing M3U8 stream from {url}...")
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     temp_file.close()
#     try:
#         cmd = [
#             "ffmpeg",
#             "-i", url,
#             "-c", "copy",
#             "-bsf:a", "aac_adtstoasc",
#             "-avoid_negative_ts", "make_zero",  # Tránh timestamp âm
#             "-y",
#             temp_file.name
#         ]
#
#         print(f"🔧 Running ffmpeg command: {' '.join(cmd)}")
#
#         if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
#             raise Exception("Downloaded file is empty or doesn't exist")
#
#         print(f"✅ M3U8 stream downloaded successfully: {temp_file.name}")
#         return temp_file.name
#
#     except Exception as e:
#         if os.path.exists(temp_file.name):
#             os.remove(temp_file.name)
#         raise Exception(f"Failed to download M3U8 stream: {str(e)}")


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


def format_timestamp(seconds, vtt=False):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    sep = '.' if vtt else ','
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millisecs:03d}"


# Handle
def process_video_transcription(audio: str | BinaryIO | ndarray, file_name: str, language: str, config: ModelConfig) -> \
Dict[str, any]:
    """Enhanced background task với anti-hallucination"""
    try:
        output_dir = os.path.join("Output", file_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{file_name}.vtt")
        # Thêm file TXT output
        txt_output_file = os.path.join(output_dir, f"{file_name}.txt")
        # Load model với cấu hình tối ưu cho CPU
        model = WhisperModel(
            config.model,
            device="cpu",  # Force CPU vì server không có GPU
            compute_type="int8",  # Tối ưu cho CPU
            num_workers=4  # Sử dụng 4 cores
        )
        # Cấu hình VAD nếu được bật
        vad_parameters = None
        if config.enable_vad:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 2000,
                "speech_pad_ms": 400
            }
        # Transcribe với các tham số chống hallucination
        segments, info = model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=config.beam_size,
            patience=config.patience,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            temperature=config.temperature,
            compression_ratio_threshold=config.compression_ratio_threshold,
            log_prob_threshold=config.log_prob_threshold,
            no_speech_threshold=config.no_speech_threshold,
            condition_on_previous_text=config.condition_on_previous_text,
            initial_prompt="Đây là một video học thuật tiếng Việt về khoa học." if language == "vi" else None,
            vad_filter=config.enable_vad,
            vad_parameters=vad_parameters,
            word_timestamps=True  # Enable để có thể phân tích tốt hơn
        )
        # Filter và clean segments
        data = []
        hallucination_count = 0
        for seg in segments:
            print("id:",seg.id," Text:",seg.text," Segment:", round(seg.start, 1), round(seg.end, 1), seg.text)
            data.append({
                "id": seg.id,
                "text": seg.text,
                "start": round(seg.start, 1),
                "end": round(seg.end, 1),
            })
        # Lưu ra file JSON
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("info", info)

        # for segment in segments:
        #     if validate_segment_quality(segment):
        #         # Clean text trước khi thêm
        #         cleaned_text = clean_repetitive_text(segment.text)
        #         if cleaned_text.strip():
        #             segment.text = cleaned_text
        #             valid_segments.append(segment)
        #     else:
        #         hallucination_count += 1
        #         print(f"⚠️ Filtered hallucination: {segment.text[:50]}...")
        #
        # # Generate VTT với smart splitting
        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write("WEBVTT\n\n")
        #     for segment in valid_segments:
        #         sub_segments = smart_segment_split(segment, max_chars=60)
        #         for sub_start, sub_end, sub_text in sub_segments:
        #             if sub_text.strip():  # Chỉ ghi segment có nội dung
        #                 f.write(f"{format_timestamp(sub_start, vtt=True)} --> {format_timestamp(sub_end, vtt=True)}\n")
        #                 f.write(f"{sub_text}\n\n")
        #
        # # Generate TXT file với transcript sạch (cho RAG pipeline)
        # with open(txt_output_file, "w", encoding="utf-8") as f:
        #     # Chỉ ghi transcript thuần túy, không có metadata
        #     transcript_parts = []
        #     for segment in valid_segments:
        #         cleaned_text = clean_repetitive_text(segment.text).strip()
        #         if cleaned_text:
        #             transcript_parts.append(cleaned_text)
        #
        #     # Ghi toàn bộ transcript thành một đoạn văn liên tục
        #     f.write(" ".join(transcript_parts))
        #
        # # Cleanup
        # if os.path.exists(input_file):
        #     os.remove(input_file)
        #
        result = {
            "vtt_file_path": output_file,
            "txt_file_path": txt_output_file,
            "stats": {
                "total_segments": len(list(segments)) if segments else 0,
                "valid_segments": len(data),
                "filtered_hallucinations": hallucination_count,
                "detected_language": info.language if hasattr(info, 'language') else language,
                "language_probability": info.language_probability if hasattr(info, 'language_probability') else None
            }
        }
        return result
    except Exception as e:
        (
            print(f"❌ Failed: {e}"))
    return None
if __name__ == '__main__':
    # # input_file = "test.mp4"
    # # subprocess.run(["ffmpeg", "-i", input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file])
    #
    output_file = "audio/audio.wav"
    # # extract audio
    # # "-c", "copy",
    # # "-bsf:a", "aac_adtstoasc",
    # # "-avoid_negative_ts", "make_zero",  # Tránh timestamp âm
    # # "-y",
    config = ModelConfig()
    # # now pass the wav file
    result = process_video_transcription(
        audio=output_file,
        file_name="test",
        language="vi",
        config=config
    )
    time = 20.46
    print(round(time, 1))

