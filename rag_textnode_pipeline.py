"""
rag_textnode_pipeline.py
========================
Chuyển đổi transcript .txt (ASR tiếng Việt) thành TextNode JSON tối ưu cho RAG.
Domain-agnostic: hoạt động với mọi loại file .txt, không giới hạn môn học.

Chạy:
    python3 rag_textnode_pipeline.py \
        --input-root file_transcript \
        --output-root file_textnodes \
        --llm-proxy-url http://127.0.0.1:5000 \
        --llm-model Qwen/Qwen3-8B-AWQ \
        --include-provenance \
        --quality-min-words 28 \
        --quality-min-overlap 0.52 \
        --quality-min-unique-ratio 0.30
"""

import argparse
import json
import os
import re
import time
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = {"Theory", "Example", "Process"}

# Dedupe threshold thấp hơn (0.76 thay vì 0.88) để bắt near-duplicate
# sau khi LLM đã rewrite — đây là nguyên nhân gây duplicate trong output cũ
DEFAULT_DEDUPE_JACCARD_THRESHOLD = 0.76
DEDUPE_NGRAM_THRESHOLD = 0.82   # char-ngram threshold phụ
DEDUPE_TOKEN_SOFT = 0.65        # token threshold để kích hoạt ngram check


# ---------------------------------------------------------------------------
# FILTER PATTERNS — domain-agnostic, chỉ dùng cho meta-discourse / filler
# Match trên chuỗi đã normalize_for_match (stripped accent, lowercase)
# ---------------------------------------------------------------------------

_FILLER_PATTERNS_NORM = [
    r"\bxin chao\b",
    r"\bcam on\b",
    r"\bhen gap lai\b",
    r"\bthua cac ban\b",
    r"\bnhu chung ta da biet\b",
    r"\bv\s*v\b",
]

_LOW_INFO_SUBSTRINGS_NORM = [
    "sau nay co dieu kien chung ta se",
    "chung toi dung lai o day",
    "chung toi dung o day xin",
    "hen gap lai cac ban",
    "xin cam on cac ban",
    "trong buoi trao doi hom nay",
    "chung toi se tiep tuc o chuyen de",
    "tu nhien toi lai nho den",
    "bong chan toi lai nho",
]

_ORPHAN_PREFIXES_NORM = [
    "cung de cap den",
    "noi cach khac",
    "nhung ma",
    "vi vay",
    "tuy nhien",
    "nhu vay",
    "do do",
    "nhu da noi",
    "nhu toi da",
    "thua cac ban",
    "toi nho den",
]

_META_DISCOURSE_PATTERNS_NORM = [
    r"\bthua cac ban\b",
    r"\btoi nho den\b",
    r"\btu nhien toi\b",
    r"\bchung toi gioi thieu\b",
    r"\bchung toi se tiep tuc\b",
    r"\btrong buoi trao doi\b",
    r"\bbong chan toi\b",
    r"\bchung ta chuyen sang\b",
    r"\bchung ta da tim hieu\b",
    r"\btiep theo chung ta\b",
    r"\bnhu da trinh bay\b",
    r"\bnhu toi da noi\b",
    # Bổ sung: pattern lịch học / hướng dẫn học tập
    r"\btrong qua trinh hoc tap co mot chuyen de\b",
    r"\bkhi den chuyen de do\b",
    r"\bchung ta se trao doi ky\b",
    r"\bvoi tu gio den het chuong trinh\b",
    r"\bchung ta trao doi voi nhau o cap do\b",
    r"\bhen gap lai\b",
    r"\bxin cam on\b",
]

_INCOMPLETE_TAILS_NORM = [
    "noi chung la", " va", " nhung", " hoac", " thi", " do la",
]


# ---------------------------------------------------------------------------
# STOPWORDS — tiếng Việt + English, domain-agnostic
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    # Vietnamese
    "la", "va", "cua", "cho", "voi", "trong", "nhung", "duoc", "mot", "cac",
    "cung", "roi", "theo", "nay", "kia", "nhieu", "nhat", "nhu", "khi",
    "neu", "thi", "do", "ay", "noi", "chung", "de", "chi", "phan", "nao",
    "thoi", "dang", "se", "da", "qua", "rat", "van", "con", "nguoi",
    "moi", "hay", "hoac", "them", "deu", "lieu", "tren", "duoi", "sau",
    "truoc", "giua", "ben", "cuoi", "bat", "dau", "hai", "ba", "bon", "nam",
    "sau", "bay", "tam", "chin", "muoi",
    # English
    "the", "this", "that", "with", "from", "into", "about", "your",
    "have", "will", "been", "were", "they", "their", "there", "when",
    "which", "would", "could", "should", "also", "more", "some", "than",
    "then", "each", "both", "such", "only", "very", "just", "over",
    "after", "before", "these", "those", "being", "other", "while",
    "what", "where", "how", "why", "who", "and", "but", "for", "not",
    "are", "was", "its", "our", "can", "all", "has", "had", "may",
})


# ---------------------------------------------------------------------------
# TERM NORMALIZATION — lỗi ASR tiếng Việt phổ biến
# Không hardcode từ khóa domain — để LLM xử lý trong prompt
# ---------------------------------------------------------------------------

TERM_NORMALIZATION: dict[str, str] = {
    "chi thuc": "tri thuc",
    "chiet hoc": "triet hoc",
    "trinh the": "chinh the",
    "clip classroom": "flip classroom",
    "mac le nin": "marx-lenin",
    "mac lennin": "marx-lenin",
    "maclenin": "marx-lenin",
    "magnin": "marx-lenin",
    "mark denny": "marx-lenin",
    "maclean": "marx-lenin",
    "marx lenin": "marx-Lenin",
}


# ---------------------------------------------------------------------------
# PROMPTS — domain-agnostic
# Không đề cập môn học cụ thể trong quy tắc chunking
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Bạn là chuyên gia trích xuất dữ liệu RAG từ transcript bài giảng tiếng Việt. "
    "Nhiệm vụ: chuyển đổi transcript nói thành TextNode JSON học thuật, "
    "độc lập, không trùng lặp. Chỉ trả về JSON array, không giải thích."
)

USER_PROMPT_TEMPLATE = """\
### NHIỆM VỤ
Trích xuất các đơn vị tri thức (TextNode) từ đoạn transcript bài giảng nói dưới đây.

### BƯỚC 1 — CHUYỂN VĂN NÓI SANG VĂN VIẾT
Transcript được tạo bằng ASR tiếng Việt nên có nhiều đặc điểm văn nói:
- XÓA: từ lặp, filler ("ờ", "à", "thì là", "cái này là"), lời chào/kết bài.
- XÓA: câu meta của giảng viên ("bây giờ chúng ta chuyển sang", "như tôi đã nói", "thưa các bạn").
- CHUẨN HÓA: câu chưa hoàn chỉnh → câu hoàn chỉnh; thêm dấu câu còn thiếu.
- THAY THẾ: đại từ mơ hồ ("nó", "điều ấy", "vấn đề này") → danh từ cụ thể theo ngữ cảnh.
- SỬA lỗi ASR thường gặp: "chi thức"→"tri thức", "trinh thể"→"chỉnh thể".

### BƯỚC 2 — ATOMIC CHUNKING (QUY TẮC QUAN TRỌNG NHẤT)
**MỖI NODE = ĐÚNG 1 Ý CHÍNH DUY NHẤT.**

Nguyên tắc tách theo đơn vị ngữ nghĩa:
- 1 định nghĩa = 1 node. Nếu đoạn có N định nghĩa khác nhau → tạo N node riêng.
- 1 đặc trưng / tính chất / thuộc tính = 1 node.
- 1 ví dụ minh họa độc lập và đủ dài = 1 node riêng.
- 1 cặp so sánh / đối lập = 1 node.

KHÔNG GOM các ý khác chủ đề vào 1 node dù chúng liền kề trong transcript.
KHÔNG TẠO node mới nếu nội dung chỉ lặp lại điều đã nói ở node trước — hãy dừng.
KHÔNG THÊM thông tin không có trong transcript (không suy diễn, không bịa).

### BƯỚC 3 — TIÊU CHUẨN CHẤT LƯỢNG NODE
- Độ dài: 60–350 từ. Tự giải thích được khi đọc độc lập, không cần context.
- Chỉ giữ: định nghĩa, lý thuyết, ví dụ cụ thể, logic phân tích, số liệu thực tế.
- Bỏ: thông tin cá nhân giảng viên, lịch học, câu hỏi tu từ, hướng dẫn học tập chung.

### BƯỚC 4 — CHUẨN HÓA CÔNG THỨC VẬT LÝ SANG LATEX (BẮT BUỘC NẾU CÓ CÔNG THỨC)
- Nếu transcript có công thức/toán học, hãy chuẩn hóa về ký hiệu chuẩn và ghi ở dạng LaTeX.
- Ví dụ: "omega bình phương" -> "\\omega^2", "căn k trên m" -> "\\sqrt{k/m}",
  "v bằng trừ omega a sin" -> "v = -\\omega A\\sin(\\omega t + \\varphi)".
- Chỉ chuẩn hóa những công thức thực sự có trong transcript, KHÔNG tự bịa hoặc suy diễn thêm.
- Nếu không chắc ký hiệu, giữ nguyên diễn giải bằng chữ trong `formula_text` và để `formula_latex` là chuỗi rỗng.
- Không dùng markdown code fence cho công thức; chỉ ghi chuỗi LaTeX thuần.

### BƯỚC 5 — QA AUGMENTATION (BẮT BUỘC cho mỗi node)
Tạo 2–3 câu hỏi mà node đó có thể trả lời trực tiếp:
- Đa dạng dạng: định nghĩa ("X là gì?"), giải thích ("Tại sao X?"), so sánh ("X khác Y thế nào?").
- KHÔNG đặt câu hỏi mà node không trả lời được.
- Câu hỏi phải tự nhiên như người học thật sự sẽ hỏi.

### OUTPUT — CHỈ TRẢ VỀ JSON ARRAY THUẦN TÚY
[
  {{
    "text": "Nội dung học thuật đã chuẩn hóa từ văn nói sang văn viết...",
    "metadata": {{
      "subject": "",
      "page": null,
      "topic": "Tên chủ đề ngắn gọn và cụ thể",
      "category": "Theory",
      "keywords": ["thuật ngữ chuyên môn 1", "thuật ngữ 2", "thuật ngữ 3"],
      "has_code": false,
      "file_name": "",
            "formulas": [
                {{
                    "formula_text": "Mô tả công thức bằng chữ",
                    "formula_latex": "\\omega = \\sqrt{k/m}",
                    "symbols": ["\\omega: tần số góc", "k: độ cứng lò xo", "m: khối lượng"]
                }}
            ],
      "question_templates": [
        "Câu hỏi 1?",
        "Câu hỏi 2?",
        "Câu hỏi 3?"
      ]
    }}
  }}
]
Lưu ý: category CHỈ được là Theory | Example | Process

### TRANSCRIPT CẦN XỬ LÝ:
<<<
{CHUNK_TEXT}
>>>
"""

_FORCE_MULTI_SUFFIX = (
    "\n\n### RÀNG BUỘC BỔ SUNG"
    "\n- Transcript chứa nhiều luận điểm — PHẢI tách thành nhiều node riêng biệt."
    "\n- KHÔNG gom các chủ đề khác nhau vào 1 node."
    "\n- Ưu tiên tạo 3–6 node có chủ đề tách biệt rõ ràng."
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    input_root: Path
    output_root: Path
    llm_proxy_url: str
    llm_model: str
    llm_timeout_seconds: int
    llm_max_tokens: int
    llm_temperature: float
    input_chunk_max_words: int
    input_chunk_min_words: int
    input_chunk_overlap_words: int
    quality_min_words: int
    quality_max_words: int
    quality_min_overlap: float
    quality_min_unique_ratio: float
    include_provenance: bool
    dedupe_threshold: float = field(default=DEFAULT_DEDUPE_JACCARD_THRESHOLD)
    file_max_retries: int = field(default=2)
    file_retry_backoff_seconds: float = field(default=2.0)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def resolve_llm_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions") or base.endswith("/chat/completions"):
        return [base]
    return [
        f"{base}/v1/chat/completions",
        f"{base}/chat/completions",
        f"{base}/chat",
    ]


def extract_json_array(text: str) -> list[dict[str, Any]] | None:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return None

    candidate = cleaned[start: end + 1]
    for attempt in [candidate, re.sub(r",\s*([\]}])", r"\1", candidate)]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def call_llm_for_textnodes(
    chunk_text: str,
    cfg: PipelineConfig,
    *,
    force_multi_nodes: bool = False,
) -> list[dict[str, Any]]:
    user_prompt = USER_PROMPT_TEMPLATE.replace("{CHUNK_TEXT}", chunk_text)
    if force_multi_nodes:
        user_prompt += _FORCE_MULTI_SUFFIX

    last_error: str | None = None
    for endpoint in resolve_llm_urls(cfg.llm_proxy_url):
        try:
            resp = requests.post(
                endpoint,
                json={
                    "model": cfg.llm_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": cfg.llm_temperature,
                    "max_tokens": cfg.llm_max_tokens,
                    "top_p": 0.9,
                },
                timeout=cfg.llm_timeout_seconds,
            )
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                continue

            payload = resp.json()
            assistant_text = ""
            if "choices" in payload:
                assistant_text = (
                    payload.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            elif "response" in payload:
                assistant_text = str(payload.get("response", ""))
            elif "text" in payload:
                assistant_text = str(payload.get("text", ""))

            parsed = extract_json_array(assistant_text)
            if parsed is not None:
                return parsed
            last_error = "LLM response did not contain a valid JSON array"
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)

    raise RuntimeError(f"LLM call failed: {last_error or 'unknown error'}")


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.?!])\1+", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    return normalize_spaces(text)


def strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_terms(text: str) -> str:
    out = text
    for wrong, right in TERM_NORMALIZATION.items():
        out = re.sub(rf"\b{re.escape(wrong)}\b", right, out, flags=re.IGNORECASE)
    return out


def normalize_for_match(text: str) -> str:
    """Chuẩn hóa để so sánh: lowercase, bỏ dấu, giữ alphanumeric + space."""
    s = normalize_terms(text.lower())
    s = strip_accents(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return normalize_spaces(s)


# ---------------------------------------------------------------------------
# KEYWORD EXTRACTION — bigram-aware cho tiếng Việt compound words
# ---------------------------------------------------------------------------

def _extract_bigrams(tokens: list[str]) -> list[str]:
    """
    Tạo bigram từ list token.
    Bỏ bigram nếu bất kỳ token nào là stopword hoặc quá ngắn.
    """
    result = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if (
            a not in STOPWORDS and b not in STOPWORDS
            and len(a) >= 2 and len(b) >= 2
        ):
            result.append(f"{a} {b}")
    return result


def infer_keywords(text: str, min_count: int = 3, max_count: int = 5) -> list[str]:
    """
    Trích từ khóa domain-agnostic từ text.
    Ưu tiên bigram (compound words) trước unigram.
    Giải quyết vấn đề cũ: "tri thức" bị split thành "tri" và "thức".
    """
    norm = normalize_for_match(text)
    tokens = [
        t for t in norm.split()
        if len(t) >= 2 and t not in STOPWORDS and not t.isdigit()
    ]

    # Đếm bigram
    bigram_freq: dict[str, int] = {}
    for bg in _extract_bigrams(tokens):
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

    # Đếm unigram (chỉ >= 3 chars để tránh fragment)
    unigram_freq: dict[str, int] = {}
    for t in tokens:
        if len(t) >= 3:
            unigram_freq[t] = unigram_freq.get(t, 0) + 1

    picked: list[str] = []
    seen: set[str] = set()

    # Bigram trước
    for bg, _ in sorted(bigram_freq.items(), key=lambda x: -x[1]):
        if len(picked) >= max_count:
            break
        parts = bg.split()
        if any(p in seen for p in parts):
            continue
        picked.append(bg)
        seen.update(parts)

    # Unigram bổ sung
    for t, _ in sorted(unigram_freq.items(), key=lambda x: (-x[1], x[0])):
        if len(picked) >= max_count:
            break
        if t not in seen:
            picked.append(t)
            seen.add(t)

    # Fallback
    if len(picked) < min_count:
        for t in tokens:
            if len(picked) >= min_count:
                break
            if t not in seen and len(t) >= 3:
                picked.append(t)
                seen.add(t)

    return picked[:max_count]


# ---------------------------------------------------------------------------
# TOKEN / SIMILARITY
# ---------------------------------------------------------------------------

def token_set(text: str) -> set[str]:
    return set(normalize_for_match(text).split())


def unique_token_ratio(text: str) -> float:
    words = normalize_for_match(text).split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def lexical_overlap_ratio(text: str, source: str) -> float:
    a = token_set(text)
    b = token_set(source)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


def char_ngram_set(text: str, n: int = 4) -> set[str]:
    s = normalize_for_match(text).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i: i + n] for i in range(len(s) - n + 1)}


# ---------------------------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    normalized = normalize_spaces(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    results = [p.strip() for p in parts if p.strip()]
    # Fallback cho transcript không có dấu chấm câu
    if len(results) <= 1 and len(normalized.split()) > 80:
        parts = re.split(r"[,;:]\s+", normalized)
        results = [p.strip() for p in parts if p.strip()]
    return results or [normalized]


# ---------------------------------------------------------------------------
# FILTERING
# ---------------------------------------------------------------------------

def is_low_info_sentence(sentence: str) -> bool:
    norm = normalize_for_match(sentence)
    if not norm:
        return True
    for pattern in _FILLER_PATTERNS_NORM:
        if re.search(pattern, norm):
            return True
    for sub in _LOW_INFO_SUBSTRINGS_NORM:
        if sub in norm:
            return True
    return False


def remove_repeated_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for s in sentences:
        key = normalize_for_match(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def clean_node_text(text: str) -> str:
    text = normalize_terms(normalize_spaces(text))
    sents = split_sentences(text)
    filtered = [s for s in sents if not is_low_info_sentence(s)] or sents
    filtered = remove_repeated_sentences(filtered)
    return normalize_punctuation(normalize_spaces(" ".join(filtered)))


def repetition_ratio(text: str) -> float:
    words = normalize_for_match(text).split()
    if len(words) < 12:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def is_incomplete_ending(text: str) -> bool:
    norm = normalize_for_match(text)
    return any(norm.endswith(t) for t in _INCOMPLETE_TAILS_NORM)


def is_orphan_phrase(text: str) -> bool:
    norm = normalize_for_match(text)
    if len(norm.split()) < 30 and any(norm.startswith(p) for p in _ORPHAN_PREFIXES_NORM):
        return True
    return norm.endswith("trong") or norm.endswith("nhu the")


def is_meta_discourse(text: str) -> bool:
    norm = normalize_for_match(text)
    return any(re.search(p, norm) for p in _META_DISCOURSE_PATTERNS_NORM)


def should_drop_node(text: str) -> bool:
    norm = normalize_for_match(text)
    if not norm:
        return True
    words = norm.split()
    if len(words) <= 10:
        return True
    if len(words) <= 18 and not re.search(r"[.!?]", text):
        return True
    if repetition_ratio(text) > 0.42:
        return True
    if is_incomplete_ending(text):
        return True
    if is_orphan_phrase(text):
        return True
    if is_meta_discourse(text):
        return True
    for pattern in _FILLER_PATTERNS_NORM:
        if re.search(pattern, norm) and len(words) < 20:
            return True
    return False


def contains_code(text: str) -> bool:
    if re.search(r"\b(def|class|return|for\s+\w+\s+in|if\s+\w+|import)\b", text):
        return True
    if re.search(r"[{}<>]=?|==|!=", text) and re.search(r"\b[A-Za-z_]\w*\s*\(", text):
        return True
    return False


# ---------------------------------------------------------------------------
# QUALITY GATE
# ---------------------------------------------------------------------------

def quality_gate_node(
    node_text: str,
    source_chunk: str,
    *,
    min_words: int,
    max_words: int,
    min_overlap: float,
    min_unique_ratio: float,
) -> bool:
    words = node_text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if unique_token_ratio(node_text) < min_unique_ratio:
        return False
    if lexical_overlap_ratio(node_text, source_chunk) < min_overlap:
        return False
    return True


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

def best_source_quote(node_text: str, source_text: str, top_k: int = 3) -> str:
    """
    Tìm đoạn trong source_text khớp nhất với node_text.

    Dùng sliding window (cửa sổ top_k câu liên tiếp) thay vì
    chọn các câu rời rạc — tránh trường hợp mọi node trong cùng
    chunk đều kéo về cùng 1 đoạn phổ biến.

    Score = precision (token node xuất hiện trong cửa sổ)
            * idf_boost (thưởng token hiếm, phạt token phổ biến)
    """
    sents = split_sentences(source_text)
    if not sents:
        return ""

    node_tokens = token_set(node_text) - STOPWORDS
    if not node_tokens:
        return ""

    # Tính IDF đơn giản: token xuất hiện trong ít câu hơn → weight cao hơn
    doc_freq: dict[str, int] = {}
    for s in sents:
        for t in token_set(s) - STOPWORDS:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    total = max(1, len(sents))

    def idf(t: str) -> float:
        return 1.0 / (doc_freq.get(t, 1) ** 0.5)

    node_weight = sum(idf(t) for t in node_tokens)
    if node_weight == 0:
        return ""

    # Sliding window: score mỗi cửa sổ top_k câu liên tiếp
    best_score = 0.0
    best_window: list[str] = []

    window_size = min(top_k, len(sents))
    for i in range(len(sents) - window_size + 1):
        window = sents[i: i + window_size]
        window_tokens = set()
        for s in window:
            window_tokens |= token_set(s) - STOPWORDS

        # Weighted precision: tỷ lệ token node có IDF cao xuất hiện trong window
        hit_weight = sum(idf(t) for t in node_tokens if t in window_tokens)
        score = hit_weight / node_weight

        if score > best_score:
            best_score = score
            best_window = window

    if not best_window or best_score < 0.1:
        return ""

    return normalize_spaces(" ".join(best_window))[:700]


# ---------------------------------------------------------------------------
# VALIDATION — QA templates
# ---------------------------------------------------------------------------

def validate_question_templates(questions: Any, node_text: str) -> list[str]:
    if not isinstance(questions, list):
        return []
    node_tokens = token_set(node_text) - STOPWORDS
    valid: list[str] = []
    seen_q: set[str] = set()
    for q in questions:
        if not isinstance(q, str):
            continue
        q = normalize_spaces(q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"
        if len(q) < 10 or len(q) > 250:
            continue
        q_norm = normalize_for_match(q)
        if q_norm in seen_q:
            continue
        seen_q.add(q_norm)
        q_tokens = token_set(q) - STOPWORDS
        if not (q_tokens & node_tokens):
            continue
        valid.append(q)
        if len(valid) >= 3:
            break
    return valid


# ---------------------------------------------------------------------------
# KEYWORD FRAGMENT DETECTION — loại ASR noise như "oi tuong", "no khong"
# ---------------------------------------------------------------------------

# Tiếng Việt: từ hợp lệ thường có cấu trúc CV hoặc CVC với nguyên âm dài
# ASR noise thường là: ghép ngẫu nhiên syllable không thành từ thật
# Approach: token stripped-accent phải khớp pattern từ tiếng Việt/Anh hợp lệ

# Pattern từ tiếng Anh hợp lệ (>= 3 chars, có trong tập từ phổ biến)
_EN_VALID_PREFIXES = frozenset({
    "abs", "acc", "act", "add", "adm", "ago", "aid", "aim", "air", "all",
    "ana", "and", "app", "arc", "are", "art", "ask", "ass", "aud", "aug",
    "bas", "beh", "bio", "bus", "cap", "cat", "cen", "cha", "che", "chi",
    "cla", "col", "com", "con", "cor", "cri", "cul", "dat", "def", "dep",
    "des", "dev", "dia", "dif", "dir", "dis", "doc", "dom", "dyn", "eco",
    "edu", "ele", "emp", "eng", "env", "est", "eth", "eve", "evo", "exa",
    "exc", "exp", "ext", "fac", "far", "fie", "fin", "for", "fra", "fun",
    "gen", "geo", "gov", "gro", "gui", "hab", "hea", "his", "hum", "hyp",
    "ide", "ima", "imp", "inc", "ind", "inf", "ins", "int", "inv", "iso",
    "jus", "kno", "lan", "law", "lea", "leg", "lib", "lin", "log", "man",
    "mar", "mat", "mea", "mec", "med", "met", "mig", "mod", "mol", "mon",
    "mor", "mot", "mul", "nat", "net", "neu", "nor", "obj", "obs", "off",
    "ont", "opt", "ord", "org", "ori", "out", "par", "pat", "per", "phi",
    "pho", "phy", "pla", "pol", "pos", "pre", "pri", "pro", "psy", "pub",
    "qua", "ran", "rat", "rec", "red", "ref", "rel", "rep", "res", "rev",
    "sci", "sec", "sel", "sem", "ser", "soc", "sol", "spe", "sta", "str",
    "sub", "sys", "tax", "tec", "the", "the", "thr", "tra", "tri", "typ",
    "und", "uni", "use", "val", "var", "ver", "vis", "voc", "war", "wor",
})

# Tiếng Việt: syllable hợp lệ sau khi strip accent phải match pattern này
# [phụ âm đầu tùy chọn] + [nguyên âm] + [phụ âm cuối tùy chọn]
_VN_SYLLABLE_RE = re.compile(
    r'^(b|c|ch|d|đ|g|gh|gi|h|k|kh|l|m|n|ng|ngh|nh|p|ph|qu|r|s|t|th|tr|v|x|z)?'
    r'(a|ă|â|e|ê|i|o|ô|ơ|u|ư|y|ai|ao|au|ay|âu|âi|eo|eu|ia|ie|io|iu|oa|oe|oi|ôi|ơi|ua|ue|ui|uo|uô|ươ|uy|ya|ye)'
    r'(c|ch|m|n|ng|nh|p|t)?$'
)

def _is_asr_noise_token(token: str) -> bool:
    """
    Phát hiện token là ASR noise sau khi đã strip accent.
    Token hợp lệ phải là syllable tiếng Việt hoặc prefix tiếng Anh nhận ra được.
    """
    t = token.strip().lower()
    if len(t) <= 1:
        return True
    # Không có nguyên âm → không phải từ
    if not re.search(r'[aeiouaăâêôơư]', t):
        return True
    # Tiếng Anh: prefix phổ biến
    if len(t) >= 3 and t[:3] in _EN_VALID_PREFIXES:
        return False
    # Tiếng Việt: kiểm tra từng syllable
    syllables = t.split()
    for syl in syllables:
        if _VN_SYLLABLE_RE.match(syl):
            return False
    # Nếu ngắn và không match pattern nào → nghi noise
    if len(t) <= 6:
        return True
    return False


def _is_valid_keyword(kw: str) -> bool:
    """
    Keyword hợp lệ: từ/cụm từ tiếng Việt hoặc tiếng Anh thật sự.
    Loại ASR noise như "oi tuong", "hoc vao", "gioi xa".
    """
    # Check trên text gốc (giữ dấu)
    kw_stripped = kw.strip()
    if not kw_stripped or len(kw_stripped.replace(" ", "")) < 2:
        return False

    norm = normalize_for_match(kw_stripped)  # strip accent
    if not norm or len(norm.replace(" ", "")) < 3:
        return False

    parts = norm.split()

    # Loại pure stopword
    if all(p in STOPWORDS for p in parts):
        return False

    # Phải có ít nhất 1 part không phải stopword và không phải noise
    meaningful = [p for p in parts if p not in STOPWORDS]
    if not meaningful:
        return False

    # Mỗi part có nghĩa phải pass noise check
    noise_count = sum(1 for p in meaningful if _is_asr_noise_token(p))
    # Nếu tất cả part có nghĩa đều là noise → loại
    if noise_count == len(meaningful):
        return False
    # Với bigram: cho phép 1 part là noise nếu part kia hợp lệ
    # Nhưng với unigram: phải pass hoàn toàn
    if len(parts) == 1 and noise_count > 0:
        return False

    return True


# ---------------------------------------------------------------------------
# VALIDATION — keywords (bigram-aware)
# ---------------------------------------------------------------------------

def validate_keywords(
    keywords_raw: list[Any],
    text: str,
    topic: str,
) -> list[str]:
    text_norm = normalize_for_match(text)
    keywords: list[str] = []

    def kw_valid(kw: str) -> bool:
        k = normalize_for_match(kw)
        if not k or len(k.replace(" ", "")) < 3:
            return False
        if all(p in STOPWORDS for p in k.split()):
            return False
        # Kiểm tra hợp lệ (loại ASR noise)
        if not _is_valid_keyword(kw):
            return False
        # Phải xuất hiện trong text
        return k in text_norm

    if isinstance(keywords_raw, list):
        for item in keywords_raw:
            token = normalize_spaces(str(item)).strip()
            if token and token not in keywords and kw_valid(token):
                keywords.append(token)

    if len(keywords) < 3:
        for t in infer_keywords(text):
            if t not in keywords:
                keywords.append(t)
            if len(keywords) >= 5:
                break

    if len(keywords) < 3:
        for t in infer_keywords(topic, min_count=1, max_count=3):
            if t not in keywords:
                keywords.append(t)
            if len(keywords) >= 3:
                break

    keywords = [normalize_terms(k) for k in keywords]
    keywords = [
        k for k in keywords
        if _is_valid_keyword(k)
        and len(normalize_for_match(k).replace(" ", "")) >= 3
        and not all(p in STOPWORDS for p in normalize_for_match(k).split())
    ]
    return keywords[:5]


# ---------------------------------------------------------------------------
# CATEGORY NORMALIZATION — domain-agnostic
# ---------------------------------------------------------------------------

def normalize_category(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in ALLOWED_CATEGORIES:
        return raw
    lowered = raw.lower()
    if any(w in lowered for w in ("example", "vi du", "minh hoa", "instance", "case")):
        return "Example"
    if any(w in lowered for w in ("process", "step", "quy trinh", "buoc", "procedure")):
        return "Process"
    return "Theory"


# ---------------------------------------------------------------------------
# NODE SANITIZATION
# ---------------------------------------------------------------------------

def _split_oversized(text: str, max_words: int = 500) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    sents = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for s in sents:
        w = len(s.split())
        if current and count + w > max_words:
            chunks.append(" ".join(current).strip())
            current, count = [s], w
        else:
            current.append(s)
            count += w
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def sanitize_node(
    raw_node: dict[str, Any],
    subject: str,
    file_name: str,
    source_chunk: str,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    if not isinstance(raw_node, dict):
        return []

    text = clean_node_text(str(raw_node.get("text", "")))
    if should_drop_node(text):
        return []
    if not quality_gate_node(
        text, source_chunk,
        min_words=cfg.quality_min_words,
        max_words=cfg.quality_max_words,
        min_overlap=cfg.quality_min_overlap,
        min_unique_ratio=cfg.quality_min_unique_ratio,
    ):
        return []

    meta = raw_node.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    topic = normalize_terms(normalize_spaces(str(meta.get("topic", ""))))
    if not topic:
        topic = split_sentences(text)[0][:120]

    category = normalize_category(meta.get("category"))
    keywords = validate_keywords(meta.get("keywords", []), text, topic)
    has_code = meta.get("has_code") if isinstance(meta.get("has_code"), bool) else contains_code(text)
    question_templates = validate_question_templates(meta.get("question_templates", []), text)

    finalized: list[dict[str, Any]] = []
    for part in _split_oversized(text):
        if should_drop_node(part):
            continue
        if not quality_gate_node(
            part, source_chunk,
            min_words=cfg.quality_min_words,
            max_words=cfg.quality_max_words,
            min_overlap=cfg.quality_min_overlap,
            min_unique_ratio=cfg.quality_min_unique_ratio,
        ):
            continue

        meta_out: dict[str, Any] = {
            "subject": subject,
            "page": None,
            "topic": topic,
            "category": category,
            "keywords": keywords,
            "has_code": has_code,
            "file_name": file_name,
            "question_templates": question_templates,
        }
        if cfg.include_provenance:
            meta_out["source_coverage"] = round(lexical_overlap_ratio(part, source_chunk), 4)
            meta_out["source_quote"] = best_source_quote(part, source_chunk)

        finalized.append({"id": str(uuid.uuid4()), "text": part, "metadata": meta_out})

    return finalized


# ---------------------------------------------------------------------------
# DEDUPLICATION — threshold thấp hơn để bắt near-duplicate post-rewrite
# ---------------------------------------------------------------------------

def dedupe_nodes(
    nodes: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_DEDUPE_JACCARD_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Loại near-duplicate bằng 3 signal kết hợp:
    1. Token Jaccard >= threshold → duplicate
    2. Token Jaccard >= DEDUPE_TOKEN_SOFT AND char-4gram Jaccard >= DEDUPE_NGRAM_THRESHOLD → duplicate
    3. Topic Jaccard >= 0.85 AND token Jaccard >= 0.55 → duplicate
       (bắt trường hợp LLM rewrite nội dung giống nhau nhưng wording khác)
    """
    seen_exact: set[str] = set()
    deduped: list[dict[str, Any]] = []
    tok_sets: list[set[str]] = []
    ngram_sets: list[set[str]] = []
    topic_sets: list[set[str]] = []

    for node in nodes:
        text = str(node.get("text", ""))
        key = normalize_for_match(text)
        if not key or key in seen_exact:
            continue

        cur_tok = set(key.split())
        cur_ng = char_ngram_set(text, n=4)
        meta = node.get("metadata", {}) or {}
        cur_topic = set(normalize_for_match(str(meta.get("topic", ""))).split()) - STOPWORDS

        is_dup = False
        for i in range(len(deduped)):
            tok_sim = jaccard_similarity(cur_tok, tok_sets[i])

            # Signal 1: token Jaccard cao
            if tok_sim >= threshold:
                is_dup = True
                break

            # Signal 2: token moderate + char ngram cao
            if tok_sim >= DEDUPE_TOKEN_SOFT:
                if jaccard_similarity(cur_ng, ngram_sets[i]) >= DEDUPE_NGRAM_THRESHOLD:
                    is_dup = True
                    break

            # Signal 3: cùng topic + token moderate
            # Bắt near-duplicate sau LLM rewrite khéo (ví dụ: node 5 vs node 8)
            if tok_sim >= 0.50 and cur_topic and topic_sets[i]:
                topic_sim = jaccard_similarity(cur_topic, topic_sets[i])
                if topic_sim >= 0.80:
                    is_dup = True
                    break

        if is_dup:
            continue

        seen_exact.add(key)
        tok_sets.append(cur_tok)
        ngram_sets.append(cur_ng)
        topic_sets.append(cur_topic)
        deduped.append(node)

    return deduped


# ---------------------------------------------------------------------------
# MERGE SHORT NODES
# ---------------------------------------------------------------------------

def merge_short_nodes_by_topic(
    nodes: list[dict[str, Any]],
    min_words: int = 28,
    max_words: int = 220,
) -> list[dict[str, Any]]:
    if len(nodes) <= 3:
        return nodes

    def topic_sim(a: str, b: str) -> float:
        sa = set(normalize_for_match(a).split()) - STOPWORDS
        sb = set(normalize_for_match(b).split()) - STOPWORDS
        return jaccard_similarity(sa, sb)

    merged: list[dict[str, Any]] = []
    for node in nodes:
        text = str(node.get("text", "")).strip()
        if not merged:
            merged.append(node)
            continue

        prev = merged[-1]
        prev_text = str(prev.get("text", "")).strip()
        pm = prev.get("metadata", {}) or {}
        cm = node.get("metadata", {}) or {}

        same_cat = str(pm.get("category", "")) == str(cm.get("category", ""))
        same_top = topic_sim(str(pm.get("topic", "")), str(cm.get("topic", ""))) >= 0.50
        pw, cw = len(prev_text.split()), len(text.split())

        if same_cat and same_top and (pw < min_words or cw < min_words) and (pw + cw) <= max_words:
            prev["text"] = normalize_punctuation(f"{prev_text}. {text}")
            merged_qt = list(dict.fromkeys(
                pm.get("question_templates", []) + cm.get("question_templates", [])
            ))[:3]
            pm["question_templates"] = merged_qt
            continue

        merged.append(node)

    return merged


# ---------------------------------------------------------------------------
# TRANSCRIPT CHUNKING
# ---------------------------------------------------------------------------

def split_transcript_for_llm(
    text: str,
    min_words: int,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        wcount = len(words)

        if wcount > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            for i in range(0, wcount, max_words):
                piece = " ".join(words[i: i + max_words]).strip()
                if piece:
                    chunks.append(piece)
            continue

        projected = current_words + wcount
        if current and projected > max_words and current_words >= min_words:
            chunks.append(" ".join(current))
            # Overlap
            overlap: list[str] = []
            oc = 0
            for prev in reversed(current):
                overlap.insert(0, prev)
                oc += len(prev.split())
                if oc >= overlap_words:
                    break
            current = overlap + [sentence]
            current_words = sum(len(s.split()) for s in current)
        else:
            current.append(sentence)
            current_words = projected

    if current:
        chunks.append(" ".join(current))

    return [normalize_spaces(c) for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# FILE PROCESSING
# ---------------------------------------------------------------------------

def detect_subject(input_root: Path, txt_path: Path) -> str:
    try:
        rel = txt_path.relative_to(input_root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass
    return txt_path.stem


def process_transcript_file(
    txt_path: Path,
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], bool]:
    content = txt_path.read_text(encoding="utf-8", errors="ignore")
    content = normalize_spaces(content)
    if not content:
        return [], False

    subject = detect_subject(cfg.input_root, txt_path)
    file_name = txt_path.name
    had_errors = False

    input_chunks = split_transcript_for_llm(
        content,
        min_words=cfg.input_chunk_min_words,
        max_words=cfg.input_chunk_max_words,
        overlap_words=cfg.input_chunk_overlap_words,
    )

    all_nodes: list[dict[str, Any]] = []
    for idx, chunk in enumerate(input_chunks, start=1):
        print(f"  - Chunk {idx}/{len(input_chunks)} ({len(chunk.split())} words)")
        try:
            raw_nodes = call_llm_for_textnodes(chunk, cfg)
        except RuntimeError as e:
            had_errors = True
            print(f"    WARNING: {e}")
            continue
        for rn in raw_nodes:
            all_nodes.extend(sanitize_node(rn, subject, file_name, chunk, cfg))

    nodes = dedupe_nodes(all_nodes, threshold=cfg.dedupe_threshold)
    nodes = merge_short_nodes_by_topic(nodes)
    nodes = dedupe_nodes(nodes, threshold=cfg.dedupe_threshold)

    # Fallback nếu transcript dài nhưng collapse về ≤1 node
    if len(nodes) <= 1 and len(content.split()) >= 500:
        print("  - Fallback: smaller chunks + force_multi_nodes")
        fb_chunks = split_transcript_for_llm(
            content,
            min_words=max(200, cfg.input_chunk_min_words // 3),
            max_words=max(450, cfg.input_chunk_max_words // 2),
            overlap_words=max(50, cfg.input_chunk_overlap_words),
        )
        rescue: list[dict[str, Any]] = []
        for idx, chunk in enumerate(fb_chunks, start=1):
            print(f"    - Fallback chunk {idx}/{len(fb_chunks)}")
            try:
                raw_nodes = call_llm_for_textnodes(chunk, cfg, force_multi_nodes=True)
            except RuntimeError as e:
                had_errors = True
                print(f"      WARNING: {e}")
                continue
            for rn in raw_nodes:
                rescue.extend(sanitize_node(rn, subject, file_name, chunk, cfg))

        hi_thr = min(0.92, cfg.dedupe_threshold + 0.12)
        rescue = dedupe_nodes(rescue, threshold=hi_thr)
        rescue = merge_short_nodes_by_topic(rescue)
        rescue = dedupe_nodes(rescue, threshold=min(0.90, cfg.dedupe_threshold + 0.10))
        if len(rescue) > len(nodes):
            nodes = rescue

    return nodes, had_errors


def detect_course_folder(input_root: Path, txt_path: Path) -> str:
    """
    Lấy tên thư mục môn học (level 1 dưới input_root).
    Ví dụ: file_transcript/Môn Triết học Mác-Lênin/Video_2/...txt
           → "Môn Triết học Mác-Lênin"
    Nếu file nằm trực tiếp trong input_root → dùng "_root"
    """
    try:
        rel = txt_path.relative_to(input_root)
    except ValueError:
        return "_root"
    return rel.parts[0] if len(rel.parts) >= 2 else "_root"


def course_output_path(output_root: Path, course_folder: str) -> Path:
    """
    Trả về path của file JSON tổng hợp cho 1 môn học.
    Ví dụ: file_textnodes4/Môn Triết học Mác-Lênin.textnodes.json
    """
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", course_folder)
    return output_root / f"{safe_name}.textnodes.json"


def write_nodes_json(path: Path, nodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)


def run_pipeline(cfg: PipelineConfig) -> None:
    txt_files = sorted(cfg.input_root.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under: {cfg.input_root}")
        return

    print(f"Found {len(txt_files)} file(s). Dedupe threshold: {cfg.dedupe_threshold}")

    # Gom file theo môn học (level-1 subfolder)
    course_files: dict[str, list[Path]] = defaultdict(list)
    for txt_path in txt_files:
        course = detect_course_folder(cfg.input_root, txt_path)
        course_files[course].append(txt_path)

    total_nodes = 0
    for course, files in sorted(course_files.items()):
        print(f"\n{'='*60}")
        print(f"Course: {course} ({len(files)} file(s))")
        print(f"{'='*60}")

        course_nodes: list[dict[str, Any]] = []
        pending_files = sorted(files)
        attempts: dict[Path, int] = defaultdict(int)

        while pending_files:
            txt_path = pending_files.pop(0)
            attempts[txt_path] += 1
            current_attempt = attempts[txt_path]

            print(
                f"\nProcessing: {txt_path} "
                f"(attempt {current_attempt}/{cfg.file_max_retries + 1})"
            )
            try:
                nodes, had_errors = process_transcript_file(txt_path, cfg)
            except Exception as exc:
                had_errors = True
                nodes = []
                print(f"  -> FAILED: {exc}")

            if had_errors:
                print("  -> File had LLM errors; discarding nodes from this attempt")
                if current_attempt <= cfg.file_max_retries:
                    if cfg.file_retry_backoff_seconds > 0:
                        time.sleep(cfg.file_retry_backoff_seconds)
                    pending_files.append(txt_path)
                    print("  -> Requeued file for full retry")
                else:
                    print("  -> Reached max retries; skipping file (no partial nodes kept)")
                continue

            course_nodes.extend(nodes)
            print(f"  -> {len(nodes)} node(s)")

        # Dedupe lần cuối toàn bộ môn (bắt duplicate giữa các file khác nhau)
        before = len(course_nodes)
        course_nodes = dedupe_nodes(course_nodes, threshold=cfg.dedupe_threshold)
        if len(course_nodes) < before:
            print(f"  [cross-file dedupe] {before} → {len(course_nodes)} nodes")

        out_path = course_output_path(cfg.output_root, course)
        write_nodes_json(out_path, course_nodes)
        total_nodes += len(course_nodes)
        print(f"\n  Course total: {len(course_nodes)} node(s) → {out_path}")

    print(f"\nDone. Total nodes across all courses: {total_nodes}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert transcript .txt files (Vietnamese ASR) "
            "to high-quality TextNode JSON for RAG. Domain-agnostic."
        )
    )
    parser.add_argument("--input-root", default="file_transcript",
                        help="Root folder containing .txt files")
    parser.add_argument("--output-root", default="file_textnodes",
                        help="Output root for TextNode JSON")
    parser.add_argument("--llm-proxy-url",
                        default=os.getenv("LLM_PROXY_URL", "http://127.0.0.1:5000"),
                        help="LLM server URL (vLLM/Ollama/OpenAI-compatible)")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--llm-timeout-seconds", type=int, default=180)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--input-chunk-min-words", type=int, default=900,
                        help="Min words per chunk sent to LLM")
    parser.add_argument("--input-chunk-max-words", type=int, default=1400,
                        help="Max words per chunk sent to LLM")
    parser.add_argument("--input-chunk-overlap-words", type=int, default=70,
                        help="Overlap words between consecutive chunks")
    parser.add_argument("--quality-min-words", type=int, default=28,
                        help="Min words for a node to pass quality gate")
    parser.add_argument("--quality-max-words", type=int, default=500,
                        help="Max words for a node to pass quality gate")
    parser.add_argument("--quality-min-overlap", type=float, default=0.52,
                        help="Min lexical overlap ratio node↔source chunk")
    parser.add_argument("--quality-min-unique-ratio", type=float, default=0.30,
                        help="Min unique token ratio to filter repetitive nodes")
    parser.add_argument(
        "--dedupe-threshold", type=float, default=DEFAULT_DEDUPE_JACCARD_THRESHOLD,
        help=(
            f"Jaccard threshold for deduplication "
            f"(default: {DEFAULT_DEDUPE_JACCARD_THRESHOLD}). "
            "Lower = more aggressive. Recommended range: 0.70–0.85."
        ),
    )
    parser.add_argument(
        "--file-max-retries", type=int, default=2,
        help=(
            "Max number of full-file retries when any LLM error occurs in that file. "
            "Total attempts = file-max-retries + 1."
        ),
    )
    parser.add_argument(
        "--file-retry-backoff-seconds", type=float, default=2.0,
        help="Wait time (seconds) before requeueing a failed file",
    )
    parser.add_argument("--include-provenance", action="store_true",
                        help="Add source_coverage + source_quote to metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        llm_proxy_url=args.llm_proxy_url,
        llm_model=args.llm_model,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        input_chunk_min_words=args.input_chunk_min_words,
        input_chunk_max_words=args.input_chunk_max_words,
        input_chunk_overlap_words=args.input_chunk_overlap_words,
        quality_min_words=args.quality_min_words,
        quality_max_words=args.quality_max_words,
        quality_min_overlap=args.quality_min_overlap,
        quality_min_unique_ratio=args.quality_min_unique_ratio,
        include_provenance=args.include_provenance,
        dedupe_threshold=args.dedupe_threshold,
        file_max_retries=max(0, args.file_max_retries),
        file_retry_backoff_seconds=max(0.0, args.file_retry_backoff_seconds),
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()