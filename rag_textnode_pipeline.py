import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ALLOWED_CATEGORIES = {"Theory", "Example", "Process"}
DEFAULT_DEDUPE_JACCARD_THRESHOLD = 0.88

FILLER_PATTERNS = [
    r"\bxin chao\b",
    r"\bcam on\b",
    r"\bhen gap lai\b",
    r"\bthua cac ban\b",
    r"\bnhu chung ta da biet\b",
    r"\bv\.v\.?\b",
]

LOW_INFO_PHRASES = [
    "sau nay co dieu kien",
    "chung toi dung lai o day",
    "chung toi dung o day",
    "hen gap lai",
    "xin cam on",
    "cung de cap den",
    "noi cach khac",
    "do la",
    "thua cac ban",
    "toi nho den",
    "tu nhien toi",
    "trong buoi trao doi",
    "chung toi se tiep tuc",
]

ORPHAN_PREFIXES = [
    "cung de cap den",
    "noi cach khac",
    "nhung ma",
    "do la",
    "vi vay",
    "tuy nhien",
    "thua cac ban",
    "toi nho den",
]

STOPWORDS = {
    "la",
    "va",
    "cua",
    "cho",
    "voi",
    "trong",
    "nhung",
    "duoc",
    "mot",
    "cac",
    "the",
    "this",
    "that",
    "with",
    "from",
    "into",
    "about",
    "your",
    "have",
    "will",
    "cung",
    "nhung",
    "roi",
    "theo",
    "nay",
    "kia",
    "mot",
    "cac",
    "trong",
    "nhieu",
    "nhat",
    "day",
    "nay",
    "kia",
    "nhu",
    "khi",
    "neu",
    "thi",
    "do",
    "ay",
    "noi",
    "chung",
    "la",
    "de",
    "duoc",
    "chi",
    "mot",
    "phan",
    "nao",
    "thoi",
    "dang",
    "se",
    "da",
    "qua",
    "rat",
    "van",
    "con",
    "nguoi",
    "nhung",
    "nhung",
    "hoc",
    "triet",
    "dinh",
    "cach",
    "van",
    "de",
    "ve",
    "moi",
}

TERM_NORMALIZATION = {
    "chiet hoc": "triet hoc",
    "mac le nin": "marx lenin",
    "mac lennin": "marx lenin",
    "maclenin": "marx lenin",
    "magnin": "marx lenin",
    "clip classroom": "flip classroom",
    "mark denny": "marx lenin",
    "maclean": "marx lenin",
    "trinh the": "chinh the",
}


SYSTEM_PROMPT_VN = """Bạn là chuyên gia trích xuất dữ liệu RAG. Nhiệm vụ của bạn là chuyển đổi transcript bài giảng thành các TextNode JSON giàu ngữ cảnh, độc lập và không trùng lặp. Chỉ trả về JSON, không giải thích."""


USER_PROMPT_TEMPLATE_VN = """### NHIỆM VỤ
Trích xuất các đơn vị tri thức (TextNode) từ đoạn transcript dưới đây.

### QUY TẮC TRÍCH XUẤT (BẮT BUỘC)
1. Gom nhóm ngữ nghĩa (Semantic Chunking):
- KHÔNG chia nhỏ các câu đơn lẻ. Gom tất cả các câu cùng giải thích cho một định nghĩa, một ví dụ hoặc một luận điểm vào MỘT node duy nhất.
- Ưu tiên các node dài và đầy đủ (200-500 từ) hơn là các node ngắn vụn vặt.

2. Tính độc lập (Self-contained context):
- Mỗi node phải tự giải thích được nội dung mà không cần đọc node trước/sau.
- THAY THẾ các đại từ mơ hồ (nó, vấn đề này, điều ấy, ông ấy) bằng danh từ cụ thể (ví dụ: triết học, Nietzsche, tính hệ thống) dựa vào ngữ cảnh của đoạn.

3. Lọc sạch dữ liệu:
- Loại bỏ hoàn toàn: lời chào, lời dẫn (như đã nói, thưa các bạn), câu thừa (vậy nhé, tiếp theo đây), và thông tin cá nhân của giảng viên.
- Chỉ giữ lại: định nghĩa, lý thuyết, ví dụ minh họa và logic phân tích.

4. Xử lý thuật ngữ và công thức:
- Chuyển các công thức toán học hoặc logic sang LaTeX ($...$ hoặc $$...$$) nếu có.
- Nếu có ví dụ (Example), phải giữ nguyên các chi tiết cụ thể để làm phong phú dữ liệu retrieval.

5. Định dạng Metadata:
- topic: tên chủ đề cụ thể (ví dụ: Tính hệ thống của tri thức triết học).
- category: chỉ chọn một trong Theory, Example, Process.
- keywords: 3-5 danh từ chuyên môn xuất hiện trong node.

### CẤU TRÚC OUTPUT (JSON)
[
    {
        "text": "Nội dung học thuật đã được làm sạch và bổ sung ngữ cảnh...",
        "metadata": {
            "subject": "",
            "page": null,
            "topic": "...",
            "category": "...",
            "keywords": ["...", "..."],
            "has_code": false,
            "file_name": ""
        }
    }
]

### TRANSCRIPT CHUNK CẦN XỬ LÝ:
<<<
{CHUNK_TEXT}
>>>
"""


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
    if start < 0 or end < 0 or end < start:
        return None

    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.?!])\1+", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    return normalize_spaces(text)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_terms(text: str) -> str:
    out = text
    for wrong, right in TERM_NORMALIZATION.items():
        out = re.sub(rf"\b{re.escape(wrong)}\b", right, out, flags=re.IGNORECASE)
    return out


def normalize_for_match(text: str) -> str:
    lowered = normalize_terms(text.lower())
    lowered = strip_accents(lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return normalize_spaces(lowered)


def token_set(text: str) -> set[str]:
    return set(normalize_for_match(text).split())


def unique_token_ratio(text: str) -> float:
    words = normalize_for_match(text).split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def lexical_overlap_ratio(text: str, source_text: str) -> float:
    a = token_set(text)
    b = token_set(source_text)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(1, len(a))


def best_source_quote(node_text: str, source_text: str, top_k: int = 2) -> str:
    source_sentences = split_sentences(source_text)
    if not source_sentences:
        return ""

    node_tokens = token_set(node_text)
    scored: list[tuple[float, str]] = []
    for sent in source_sentences:
        sent_tokens = token_set(sent)
        if not sent_tokens:
            continue
        score = len(node_tokens & sent_tokens) / max(1, len(node_tokens))
        if score > 0:
            scored.append((score, sent))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    quote = " ".join([s for _, s in scored[:top_k]])
    return normalize_spaces(quote)[:700]


def split_sentences(text: str) -> list[str]:
    normalized = normalize_spaces(text)
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    results = [p.strip() for p in parts if p.strip()]

    # Whisper TXT often lacks punctuation; split by long comma/semicolon clauses as fallback.
    if len(results) <= 1 and len(normalized.split()) > 80:
        parts = re.split(r"[,;:]\s+", normalized)
        results = [p.strip() for p in parts if p.strip()]

    if not results:
        return [normalized]
    return results


def is_low_information_sentence(sentence: str) -> bool:
    cleaned = normalize_for_match(sentence)
    if not cleaned:
        return True

    low_info_patterns = [
        r"\bxin chao\b",
        r"\bchao mung\b",
        r"\bcam on\b",
        r"\bhen gap lai\b",
        r"\bxin tam biet\b",
        r"\bthua cac ban\b",
        r"\bv\s*v\b",
        r"\bsau nay co dieu kien\b",
        r"\bchung toi dung lai o day\b",
        r"\b(phn|phan ket)\b",
    ]
    for pattern in low_info_patterns:
        if re.search(pattern, cleaned):
            return True
    return False


def remove_repeated_sentences(sentences: list[str]) -> list[str]:
    seen = set()
    output = []
    for sentence in sentences:
        key = normalize_for_match(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(sentence)
    return output


def clean_node_text(text: str) -> str:
    text = normalize_terms(normalize_spaces(text))
    sentences = split_sentences(text)
    filtered = [s for s in sentences if not is_low_information_sentence(s)]
    if not filtered:
        filtered = sentences
    filtered = remove_repeated_sentences(filtered)
    cleaned = normalize_spaces(". ".join(filtered))
    cleaned = normalize_punctuation(cleaned)
    return cleaned


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
    word_count = len(words)
    if word_count < min_words or word_count > max_words:
        return False

    if unique_token_ratio(node_text) < min_unique_ratio:
        return False

    overlap = lexical_overlap_ratio(node_text, source_chunk)
    if overlap < min_overlap:
        return False

    return True


def repetition_ratio(text: str) -> float:
    words = normalize_for_match(text).split()
    if len(words) < 12:
        return 0.0
    unique = len(set(words))
    return 1.0 - (unique / len(words))


def contains_low_info_phrase(text: str) -> bool:
    normalized = normalize_for_match(text)
    return any(phrase in normalized for phrase in LOW_INFO_PHRASES)


def is_incomplete_ending(text: str) -> bool:
    normalized = normalize_for_match(text)
    incomplete_tails = [
        "do la ve thu nhat trong",
        "noi chung la",
        "va",
        "nhung",
        "hoac",
    ]
    return any(normalized.endswith(tail) for tail in incomplete_tails)


def is_orphan_phrase(text: str) -> bool:
    normalized = normalize_for_match(text)
    words = normalized.split()
    if len(words) < 35 and any(normalized.startswith(prefix) for prefix in ORPHAN_PREFIXES):
        return True
    if normalized.endswith("trong") or normalized.endswith("nhu the"):
        return True
    return False


def is_question_fragment(text: str) -> bool:
    normalized = normalize_spaces(text)
    word_count = len(normalized.split())
    if "?" in normalized and word_count <= 40:
        return True
    if normalized.endswith("?") and word_count <= 60:
        return True
    return False


def is_conversational_style(text: str) -> bool:
    normalized = normalize_for_match(text)
    bad_patterns = [
        r"\bthua cac ban\b",
        r"\btoi nho den\b",
        r"\btu nhien toi\b",
        r"\bchung toi gioi thieu\b",
        r"\bchung toi se tiep tuc\b",
        r"\btrong buoi trao doi nay\b",
        r"\bbong chan toi\b",
    ]
    return any(re.search(pattern, normalized) for pattern in bad_patterns)


def split_transcript_for_llm(text: str, min_words: int, max_words: int, overlap_words: int) -> list[str]:
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
                current = []
                current_words = 0
            for i in range(0, wcount, max_words):
                piece_words = words[i : i + max_words]
                piece = " ".join(piece_words).strip()
                if piece:
                    chunks.append(piece)
            continue

        projected = current_words + wcount
        if current and projected > max_words and current_words >= min_words:
            chunks.append(" ".join(current))

            # Keep a semantic tail overlap to preserve cross-boundary context.
            overlap: list[str] = []
            overlap_count = 0
            for prev in reversed(current):
                overlap.insert(0, prev)
                overlap_count += len(prev.split())
                if overlap_count >= overlap_words:
                    break

            current = overlap + [sentence]
            current_words = sum(len(s.split()) for s in current)
        else:
            current.append(sentence)
            current_words = projected

    if current:
        chunks.append(" ".join(current))

    return [normalize_spaces(c) for c in chunks if c.strip()]


def call_llm_for_textnodes(
    chunk_text: str,
    cfg: PipelineConfig,
    *,
    force_multi_nodes: bool = False,
) -> list[dict[str, Any]]:
    user_prompt_VN = USER_PROMPT_TEMPLATE_VN.replace("{CHUNK_TEXT}", chunk_text)
    if force_multi_nodes:
        user_prompt_VN += (
            "\n\n### RÀNG BUỘC BỔ SUNG"
            "\n- Nếu transcript chứa nhiều luận điểm, phải tách thành nhiều node theo từng luận điểm."
            "\n- KHÔNG gom các chủ đề xa nhau vào một node duy nhất."
            "\n- Với chunk dài, ưu tiên 3-6 node có chủ đề tách biệt rõ."
        )
    last_error = None

    for endpoint in resolve_llm_urls(cfg.llm_proxy_url):
        try:
            response = requests.post(
                endpoint,
                json={
                    "model": cfg.llm_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_VN},
                        {"role": "user", "content": user_prompt_VN},
                    ],
                    "temperature": cfg.llm_temperature,
                    "max_tokens": cfg.llm_max_tokens,
                    "top_p": 0.9,
                },
                timeout=cfg.llm_timeout_seconds,
            )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                continue

            payload = response.json()
            assistant_text = ""
            if "choices" in payload:
                assistant_text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
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


def infer_keywords(text: str, min_count: int = 3, max_count: int = 5) -> list[str]:
    tokens = re.findall(r"\b\w{2,}\b", text.lower(), flags=re.UNICODE)
    freq: dict[str, int] = {}
    for token in tokens:
        base = normalize_for_match(token)
        if not base or base in STOPWORDS:
            continue
        if len(base) < 3:
            continue
        if base.isdigit():
            continue
        freq[token] = freq.get(token, 0) + 1

    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    picked = [k for k, _ in ranked[:max_count]]
    if len(picked) < min_count:
        for token in tokens:
            if token in STOPWORDS or token in picked:
                continue
            picked.append(token)
            if len(picked) >= min_count:
                break
    return picked[:max_count]


def should_drop_node(text: str) -> bool:
    cleaned = normalize_for_match(text)
    if not cleaned:
        return True

    words = cleaned.split()
    if len(words) <= 10:
        return True

    if len(words) <= 18 and not re.search(r"[.!?]", text):
        return True

    if repetition_ratio(text) > 0.42:
        return True

    if contains_low_info_phrase(text):
        return True

    if is_incomplete_ending(text):
        return True

    if is_orphan_phrase(text):
        return True

    if is_question_fragment(text):
        return True

    if is_conversational_style(text):
        return True

    for pattern in FILLER_PATTERNS:
        if re.search(pattern, cleaned):
            if len(words) < 20:
                return True

    return False


def contains_code(text: str) -> bool:
    if re.search(r"\b(def|class|return|for\s+\w+\s+in|if\s+\w+|import)\b", text):
        return True
    if re.search(r"[{}<>]=?|==|!=|\(|\)", text) and re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text):
        return True
    return False


def split_oversized_node(text: str, max_words: int = 500) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        wcount = len(sentence.split())
        if current and count + wcount > max_words:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            count = wcount
        else:
            current.append(sentence)
            count += wcount

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def normalize_category(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in ALLOWED_CATEGORIES:
        return raw
    lowered = raw.lower()
    if "example" in lowered:
        return "Example"
    if "process" in lowered or "step" in lowered:
        return "Process"
    return "Theory"


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
        text,
        source_chunk,
        min_words=cfg.quality_min_words,
        max_words=cfg.quality_max_words,
        min_overlap=cfg.quality_min_overlap,
        min_unique_ratio=cfg.quality_min_unique_ratio,
    ):
        return []

    metadata = raw_node.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    topic = normalize_spaces(str(metadata.get("topic", "")))
    if not topic:
        topic = normalize_spaces(split_sentences(text)[0])[:120]
    topic = normalize_terms(topic)

    category = normalize_category(metadata.get("category"))

    keywords_raw = metadata.get("keywords", [])
    keywords: list[str] = []
    text_match_norm = normalize_for_match(text)

    def keyword_in_text(keyword: str) -> bool:
        k = normalize_for_match(keyword)
        if not k:
            return False
        if len(k) < 3:
            return False
        if k in STOPWORDS:
            return False
        return re.search(rf"\b{re.escape(k)}\b", text_match_norm) is not None

    if isinstance(keywords_raw, list):
        for item in keywords_raw:
            token = normalize_spaces(str(item))
            if token and token not in keywords and keyword_in_text(token):
                keywords.append(token)
    if len(keywords) < 3:
        inferred = infer_keywords(text)
        for token in inferred:
            if token not in keywords:
                keywords.append(token)
            if len(keywords) >= 5:
                break
    if len(keywords) < 3:
        fallback = infer_keywords(topic, min_count=1, max_count=3)
        for token in fallback:
            if token not in keywords:
                keywords.append(token)
            if len(keywords) >= 3:
                break
    keywords = [normalize_terms(k) for k in keywords]
    keywords = [k for k in keywords if len(normalize_for_match(k)) >= 3 and normalize_for_match(k) not in STOPWORDS]
    keywords = keywords[:5]

    has_code = metadata.get("has_code")
    if not isinstance(has_code, bool):
        has_code = contains_code(text)

    finalized_nodes = []
    for part in split_oversized_node(text, max_words=500):
        if should_drop_node(part):
            continue

        if not quality_gate_node(
            part,
            source_chunk,
            min_words=cfg.quality_min_words,
            max_words=cfg.quality_max_words,
            min_overlap=cfg.quality_min_overlap,
            min_unique_ratio=cfg.quality_min_unique_ratio,
        ):
            continue

        metadata_out = {
            "subject": subject,
            "page": None,
            "topic": topic,
            "category": category,
            "keywords": keywords,
            "has_code": has_code,
            "file_name": file_name,
        }

        if cfg.include_provenance:
            metadata_out["source_coverage"] = round(lexical_overlap_ratio(part, source_chunk), 4)
            metadata_out["source_quote"] = best_source_quote(part, source_chunk)

        finalized_nodes.append(
            {
                "text": part,
                "metadata": metadata_out,
            }
        )

    return finalized_nodes


def dedupe_nodes(
    nodes: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_DEDUPE_JACCARD_THRESHOLD,
) -> list[dict[str, Any]]:
    def jaccard_similarity(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def char_ngram_set(text: str, n: int = 5) -> set[str]:
        s = normalize_for_match(text).replace(" ", "")
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(0, len(s) - n + 1)}

    seen = set()
    deduped = []
    token_sets: list[set[str]] = []
    for node in nodes:
        text = str(node.get("text", ""))
        key = normalize_for_match(text)
        if not key or key in seen:
            continue

        current_tokens = set(key.split())
        current_ngrams = char_ngram_set(text)
        is_duplicate = False
        for i, prev in enumerate(token_sets):
            token_sim = jaccard_similarity(current_tokens, prev)
            if token_sim >= threshold:
                is_duplicate = True
                break

            prev_ngrams = char_ngram_set(deduped[i].get("text", ""))
            ngram_sim = jaccard_similarity(current_ngrams, prev_ngrams)
            if token_sim >= 0.72 and ngram_sim >= 0.86:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        seen.add(key)
        token_sets.append(current_tokens)
        deduped.append(node)
    return deduped


def merge_short_nodes_by_topic(
    nodes: list[dict[str, Any]],
    min_words: int = 28,
    max_words: int = 220,
) -> list[dict[str, Any]]:
    # Avoid over-merging when node count is already low.
    if len(nodes) <= 3:
        return nodes

    if not nodes:
        return nodes

    merged: list[dict[str, Any]] = []
    def topic_similarity(a: str, b: str) -> float:
        set_a = set(normalize_for_match(a).split())
        set_b = set(normalize_for_match(b).split())
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0

    for node in nodes:
        text = str(node.get("text", "")).strip()
        if not merged:
            merged.append(node)
            continue

        prev = merged[-1]
        prev_text = str(prev.get("text", "")).strip()
        prev_words = len(prev_text.split())
        curr_words = len(text.split())

        prev_meta = prev.get("metadata", {}) if isinstance(prev.get("metadata"), dict) else {}
        curr_meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        prev_topic_raw = str(prev_meta.get("topic", ""))
        curr_topic_raw = str(curr_meta.get("topic", ""))
        prev_topic = normalize_for_match(prev_topic_raw)
        curr_topic = normalize_for_match(curr_topic_raw)
        same_category = str(prev_meta.get("category", "")) == str(curr_meta.get("category", ""))
        same_topic = prev_topic and curr_topic and (
            prev_topic == curr_topic or topic_similarity(prev_topic_raw, curr_topic_raw) >= 0.55
        )

        can_merge = same_category and same_topic and (prev_words < min_words or curr_words < min_words)
        if can_merge and (prev_words + curr_words) <= max_words:
            prev["text"] = normalize_punctuation(f"{prev_text}. {text}")
            continue

        merged.append(node)

    return merged


def detect_subject(input_root: Path, txt_path: Path) -> str:
    rel = txt_path.relative_to(input_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return txt_path.parent.name


def process_transcript_file(txt_path: Path, cfg: PipelineConfig) -> list[dict[str, Any]]:
    content = txt_path.read_text(encoding="utf-8", errors="ignore")
    content = normalize_spaces(content)
    if not content:
        return []

    subject = detect_subject(cfg.input_root, txt_path)
    file_name = txt_path.name
    input_chunks = split_transcript_for_llm(
        content,
        min_words=cfg.input_chunk_min_words,
        max_words=cfg.input_chunk_max_words,
        overlap_words=cfg.input_chunk_overlap_words,
    )

    all_nodes: list[dict[str, Any]] = []
    for idx, chunk in enumerate(input_chunks, start=1):
        print(f"  - LLM semantic chunk {idx}/{len(input_chunks)}")
        raw_nodes = call_llm_for_textnodes(chunk, cfg)
        for raw_node in raw_nodes:
            all_nodes.extend(
                sanitize_node(
                    raw_node,
                    subject=subject,
                    file_name=file_name,
                    source_chunk=chunk,
                    cfg=cfg,
                )
            )

    nodes = dedupe_nodes(all_nodes)
    nodes = merge_short_nodes_by_topic(nodes)
    nodes = dedupe_nodes(nodes)

    # Fallback pass: if long transcript collapses to <=1 node, force a multi-node extraction.
    if len(nodes) <= 1 and len(content.split()) >= 500:
        fallback_chunks = split_transcript_for_llm(
            content,
            min_words=max(260, cfg.input_chunk_min_words // 3),
            max_words=max(520, cfg.input_chunk_max_words // 2),
            overlap_words=max(60, cfg.input_chunk_overlap_words),
        )
        rescue_nodes: list[dict[str, Any]] = []
        for idx, chunk in enumerate(fallback_chunks, start=1):
            print(f"  - Fallback semantic chunk {idx}/{len(fallback_chunks)}")
            raw_nodes = call_llm_for_textnodes(chunk, cfg, force_multi_nodes=True)
            for raw_node in raw_nodes:
                rescue_nodes.extend(
                    sanitize_node(
                        raw_node,
                        subject=subject,
                        file_name=file_name,
                        source_chunk=chunk,
                        cfg=cfg,
                    )
                )

        rescue_nodes = dedupe_nodes(rescue_nodes, threshold=0.92)
        rescue_nodes = merge_short_nodes_by_topic(rescue_nodes)
        rescue_nodes = dedupe_nodes(rescue_nodes, threshold=0.90)
        if len(rescue_nodes) > len(nodes):
            nodes = rescue_nodes

    return nodes


def output_path_for_file(input_root: Path, output_root: Path, txt_path: Path) -> Path:
    rel = txt_path.relative_to(input_root)
    out_rel = rel.with_suffix(".textnodes.json")
    return output_root / out_rel


def write_nodes_json(path: Path, nodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)


def run_pipeline(cfg: PipelineConfig) -> None:
    txt_files = sorted(cfg.input_root.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt transcript found under: {cfg.input_root}")
        return

    total_nodes = 0
    for txt_path in txt_files:
        print(f"Processing: {txt_path}")
        try:
            nodes = process_transcript_file(txt_path, cfg)
            out_path = output_path_for_file(cfg.input_root, cfg.output_root, txt_path)
            write_nodes_json(out_path, nodes)
            total_nodes += len(nodes)
            print(f"  -> {len(nodes)} node(s) written to {out_path}")
        except Exception as exc:
            print(f"  -> FAILED: {exc}")

    print(f"Done. Total nodes: {total_nodes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert transcript TXT files to high-quality TextNode JSON for RAG retrieval."
    )
    parser.add_argument("--input-root", default="file_vtt", help="Root folder containing transcript .txt files")
    parser.add_argument("--output-root", default="file_textnodes", help="Output root folder for TextNode JSON")
    parser.add_argument("--llm-proxy-url", default=os.getenv("LLM_PROXY_URL", "http://127.0.0.1:5000"))
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--llm-timeout-seconds", type=int, default=180)
    parser.add_argument("--llm-max-tokens", type=int, default=3500)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--input-chunk-min-words", type=int, default=900)
    parser.add_argument("--input-chunk-max-words", type=int, default=1400)
    parser.add_argument("--input-chunk-overlap-words", type=int, default=70)
    parser.add_argument("--quality-min-words", type=int, default=35)
    parser.add_argument("--quality-max-words", type=int, default=500)
    parser.add_argument("--quality-min-overlap", type=float, default=0.62)
    parser.add_argument("--quality-min-unique-ratio", type=float, default=0.36)
    parser.add_argument("--include-provenance", action="store_true")
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
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
