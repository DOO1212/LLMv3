"""Manual search using LLM intent refinement + SQLite keyword matching."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

from config import OUTPUT_DIR, SQLITE_PATH
from config.search import compress_keyword_tokens
from llm_fallback import (
    TRANSLATION_LOCALE_ISO,
    build_translategemma_prompt,
    generate_text,
    get_manual_intent_model,
    get_translation_model,
    uses_translategemma_translation,
)

MANUAL_PDF_PATH = Path(OUTPUT_DIR) / "LG_CodeZero_A9_Air_manual_20240111.pdf"
_INDEX_READY = False

DEFAULT_EMBED_URL = os.environ.get("MANUAL_EMBED_URL", "http://127.0.0.1:11434").strip()
# 매뉴얼 벡터 검색 기본: bge-m3 (`ollama pull bge-m3`). 다른 모델은 MANUAL_EMBED_MODEL 로만 지정.
_EMBED_ENV = os.environ.get("MANUAL_EMBED_MODEL", "").strip()
DEFAULT_EMBED_MODEL = _EMBED_ENV or "bge-m3"
QUERY_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}
_MANUAL_VOCAB: list[str] | None = None
_MANUAL_RECOMMEND_VOCAB: list[str] | None = None
_MANUAL_SENTENCE_CANDIDATES: list[str] | None = None
_NGRAM_IDF_CACHE: dict[str, float] | None = None
MANUAL_STOPWORDS = {
    "제품",
    "설명서",
    "사용법",
    "방법",
    "가이드",
    "매뉴얼",
}
MANUAL_ANCHOR_STOPWORDS = {
    "청소기",
    "제품",
    "사용",
    "방법",
    "용도",
}
MANUAL_RECOMMEND_STOPWORDS = MANUAL_STOPWORDS | MANUAL_ANCHOR_STOPWORDS | {
    "안전",
    "기호",
    "의미",
    "주의",
    "기본",
    "구성품",
    "제품을",
    "위해",
    "사용할",
    "때",
}
MANUAL_SENTENCE_QUERY_STOPWORDS = {"적합"}
MANUAL_BOILERPLATE_TERMS = {
    "품질 보증",
    "보증 기간",
    "유상 서비스",
    "고객센터",
    "계약",
    "재발행",
    "영업 용도",
    "산업 용도",
    "상업 용도",
    "연구/실험 용도",
}
QUESTION_TAIL_TOKENS = {
    "알려줘",
    "알려줘요",
    "알려주세요",
    "가르쳐줘",
    "가르쳐줘요",
    "가르쳐주세요",
    "뭐야",
    "뭐에요",
    "뭐예요",
    "뭔가요",
    "무엇인가요",
}
QUESTION_SUFFIX_TOKENS = (
    "알려주세요",
    "가르쳐주세요",
    "해주세요",
    "해줘요",
    "해줘",
    "좀요",
    "쫌요",
    "입니다",
    "인가요",
    "요",
    "좀",
    "쫌",
)
KOREAN_PARTICLE_SUFFIXES = ("이", "가", "은", "는", "을", "를", "와", "과", "의", "에")
ANCHOR_EXCLUDE_TAILS = ("어", "요", "다", "네")
MANUAL_SINGLE_CHAR_TERMS = {"솔", "봉", "캡", "핸", "축"}

MANUAL_ANSWER_TEMPLATES = {
    "ko": "질문 '{query}' 관련 매뉴얼 근거를 찾았습니다 ({refs}).",
    "en": "Found manual evidence for '{query}' ({refs}).",
    "th": "พบข้อมูลจากคู่มือสำหรับ '{query}' ({refs})",
    "vi": "Đã tìm thấy nội dung hướng dẫn cho '{query}' ({refs}).",
    "id": "Ditemukan bukti manual untuk '{query}' ({refs}).",
    "ms": "Rujukan manual ditemui untuk '{query}' ({refs}).",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _clean_snippet(text: str, max_len: int = 240) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", str(text or ""))
    return _normalize(cleaned)[:max_len]


def _manual_translation_max_chars() -> int:
    try:
        return max(480, int(os.environ.get("MANUAL_TRANSLATION_MAX_CHARS", "1200")))
    except (TypeError, ValueError):
        return 1200


def _manual_translation_max_new_tokens() -> int:
    try:
        return max(256, int(os.environ.get("MANUAL_TRANSLATION_MAX_NEW_TOKENS", "512")))
    except (TypeError, ValueError):
        return 512


def _manual_snippet_max_chars() -> int:
    try:
        return max(240, int(os.environ.get("MANUAL_SNIPPET_MAX_CHARS", "520")))
    except (TypeError, ValueError):
        return 520


def _strip_sparse_digit_runs(text: str) -> str:
    """도면/단계 번호로 보이는 '1 2 3 4'·'1 2 4 5 1 2' 형태(한 자리 숫자만 공백으로 연속) 제거."""
    s = str(text or "")
    out = re.sub(r"(?<![0-9])(?:[0-9]\s+){3,}[0-9](?![0-9])", " ", s)
    return _normalize(out)


def _manual_snippet_for_result(raw: str) -> str:
    """검색 결과 스니펫: 레이아웃 번호 잔재 제거 후 길이 제한."""
    t = _strip_sparse_digit_runs(raw)
    return _clean_snippet(t, max_len=_manual_snippet_max_chars())


def _dedupe_similar_manual_hits(hits: list[dict]) -> list[dict]:
    """같은 페이지·동일 스니펫 앞부분이면 더 높은 점수만 유지."""
    if len(hits) <= 1:
        return hits
    best: dict[tuple[int, str], dict] = {}
    for h in hits:
        pg = int(h.get("page_no") or 0)
        pref = _compact(_normalize(str(h.get("snippet", ""))))[:96]
        key = (pg, pref)
        sc = float(h.get("score") or 0)
        if key not in best or sc > float(best[key].get("score") or 0):
            best[key] = h
    out = list(best.values())
    out.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return out


def _soft_truncate_translation(text: str, max_len: int) -> str:
    """번역문이 길 때 글자 단위로 자르지 않고 문장·단어 경계를 우선한다."""
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", str(text or ""))
    t = _normalize(cleaned)
    if len(t) <= max_len:
        return t
    head = t[:max_len]
    for sep in ("\n\n", "\n", ". ", "。", "• ", "; ", ": "):
        idx = head.rfind(sep)
        if idx >= int(max_len * 0.55):
            return t[: idx + len(sep)].strip()
    sp = head.rfind(" ")
    if sp >= int(max_len * 0.45):
        return (t[:sp].strip() + "…") if sp < len(t) - 1 else t[:sp].strip()
    return head.strip() + "…"


def _strip_question_suffixes(token: str) -> str:
    cleaned = str(token or "")
    changed = True
    while changed and cleaned:
        changed = False
        for suffix in QUESTION_SUFFIX_TOKENS:
            if cleaned.endswith(suffix) and len(cleaned) - len(suffix) >= 2:
                cleaned = cleaned[: -len(suffix)]
                changed = True
                break
    return cleaned


def _is_valid_manual_term(term: str) -> bool:
    token = _normalize(term)
    return len(token) >= 2 or token in MANUAL_SINGLE_CHAR_TERMS


def _normalize_manual_token(token: str) -> list[str]:
    """구어형/조사형 토큰을 일반 규칙으로 정규화한다."""
    base = _strip_question_suffixes(token)
    variants: list[str] = []

    # 조사 제거: "소리가" -> "소리"
    if base and len(base) >= 2 and base[-1] in KOREAN_PARTICLE_SUFFIXES:
        variants.append(base[:-1])
    elif base:
        variants.append(base)
    deduped: list[str] = []
    for v in variants:
        nv = _normalize(v)
        if _is_valid_manual_term(nv) and nv not in deduped and nv not in MANUAL_STOPWORDS:
            deduped.append(nv)
    return deduped


def _extract_terms(query: str) -> list[str]:
    terms: list[str] = []
    normalized = _normalize(query)
    # 질의 끝 질문형 표현을 먼저 제거해 검색 핵심어를 남긴다.
    if normalized:
        tail_pattern = "|".join(re.escape(x) for x in sorted(QUESTION_TAIL_TOKENS, key=len, reverse=True))
        normalized = re.sub(rf"(?:\s*(?:{tail_pattern}))+[\?\!\.]*\s*$", "", normalized).strip()
    for token in re.split(r"\s+", normalized):
        token = re.sub(r"[^\w가-힣]", "", token)
        if token in QUESTION_TAIL_TOKENS:
            continue
        terms.extend(_normalize_manual_token(token))
    # Treat spaced/unspaced variants as same intent.
    compact_query = re.sub(r"[^\w가-힣]", "", _compact(normalized))
    compact_stem = _strip_question_suffixes(compact_query)
    # 무공백 단일 질의에서만 compact 토큰을 사용해 과도한 결합어 유입을 막는다.
    if " " not in normalized and _is_valid_manual_term(compact_stem) and compact_stem not in terms:
        terms.append(compact_stem)
    unique = list(dict.fromkeys(terms))[:8]
    return compress_keyword_tokens(unique)


def _select_anchor_terms(raw_terms: list[str], all_terms: list[str]) -> list[str]:
    """Pick high-information anchors without hardcoded word patterns."""
    base = raw_terms or all_terms
    if not base:
        return []
    # Prefer longer terms as anchors (more specific), cap to top 2.
    unique = list(dict.fromkeys(base))
    # Exclude overlong compact tokens (often whole-query concatenation).
    unique = [t for t in unique if len(t) <= 8]
    # 구어형/어미성 토큰은 앵커에서 제외한다.
    unique = [t for t in unique if not any(t.endswith(tail) for tail in ANCHOR_EXCLUDE_TAILS)]
    # 조사형 토큰(예: 소리가/필터를)은 앵커에서 제외한다.
    unique = [t for t in unique if not (len(t) >= 3 and t[-1] in KOREAN_PARTICLE_SUFFIXES)]
    # 일반/광의 키워드는 앵커에서 제외해 실제 부품/증상 단어를 우선한다.
    unique = [t for t in unique if t not in MANUAL_ANCHOR_STOPWORDS]
    unique.sort(key=lambda x: (len(x), x), reverse=True)
    if unique:
        return unique[:2]
    # 모두 제거된 경우에만 원본에서 복구
    fallback = [t for t in list(dict.fromkeys(base)) if len(t) <= 8]
    fallback.sort(key=lambda x: (len(x), x), reverse=True)
    return fallback[:2]


def _contains_term(term: str, text_norm: str, text_compact: str) -> bool:
    return term in text_norm or term in text_compact


def _expanded_match_bonus(term: str, text_norm: str, text_compact: str) -> float:
    """정확 일치 외에 일반적인 형태 확장 일치 보너스를 부여한다."""
    if not term or len(term) < 1:
        return 0.0
    expanded_forms = [
        f"{term}형",
        f"{term}용",
        f"{term}용도",
    ]
    for form in expanded_forms:
        if form in text_norm or form in text_compact:
            return 0.5
    return 0.0


def _boilerplate_penalty(text_norm: str) -> float:
    """약관/보증 성격 문단을 완만하게 감점한다."""
    hits = sum(1 for token in MANUAL_BOILERPLATE_TERMS if token in text_norm)
    if hits >= 3:
        return 1.5
    if hits >= 2:
        return 1.0
    return 0.0


def _post_json(url: str, payload: dict, timeout: int = 60) -> dict | None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _embed_text_ollama(text: str, model: str, base_url: str) -> list[float] | None:
    normalized = _normalize(text)
    if not normalized:
        return None
    cache_key = (model, normalized)
    if cache_key in QUERY_EMBED_CACHE:
        return QUERY_EMBED_CACHE[cache_key]

    root = base_url.rstrip("/")
    payloads = [
        (f"{root}/api/embed", {"model": model, "input": normalized}),
        (f"{root}/api/embeddings", {"model": model, "prompt": normalized}),
    ]
    vector: list[float] | None = None
    for url, payload in payloads:
        result = _post_json(url, payload)
        if not result:
            continue
        emb = result.get("embedding")
        if emb and isinstance(emb, list):
            vector = [float(x) for x in emb]
            break
        embs = result.get("embeddings")
        if embs and isinstance(embs, list) and embs and isinstance(embs[0], list):
            vector = [float(x) for x in embs[0]]
            break

    if vector:
        QUERY_EMBED_CACHE[cache_key] = vector
    return vector


def _cosine_vectors(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _ensure_manual_embeddings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY,
            embed_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            vec_json TEXT NOT NULL
        )
        """
    )


def _load_chunk_embedding(
    conn: sqlite3.Connection,
    chunk_id: int,
    embed_model: str,
) -> list[float] | None:
    row = conn.execute(
        """
        SELECT vec_json, embed_model FROM manual_chunk_embeddings
        WHERE chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if not row or row["embed_model"] != embed_model:
        return None
    try:
        vec = json.loads(row["vec_json"])
        if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
            return [float(x) for x in vec]
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def _save_chunk_embedding(
    conn: sqlite3.Connection,
    chunk_id: int,
    embed_model: str,
    vec: list[float],
) -> None:
    conn.execute(
        """
        INSERT INTO manual_chunk_embeddings (chunk_id, embed_model, embedding_dim, vec_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            embed_model = excluded.embed_model,
            embedding_dim = excluded.embedding_dim,
            vec_json = excluded.vec_json
        """,
        (chunk_id, embed_model, len(vec), json.dumps(vec)),
    )


def _get_or_create_chunk_embedding(
    conn: sqlite3.Connection,
    chunk_id: int,
    chunk_text: str,
    embed_model: str,
    embed_url: str,
) -> list[float] | None:
    cached = _load_chunk_embedding(conn, chunk_id, embed_model)
    if cached is not None:
        return cached
    vec = _embed_text_ollama(chunk_text[:8000], embed_model, embed_url)
    if not vec:
        return None
    _save_chunk_embedding(conn, chunk_id, embed_model, vec)
    return vec


def _compose_vector_query_text(parsed: dict | None, fallback_query: str) -> str:
    if not parsed:
        return _normalize(fallback_query)
    oq = str(parsed.get("original_query") or "").strip()
    iq = str(parsed.get("intent_query") or "").strip()
    parts: list[str] = []
    if oq:
        parts.append(oq)
    if iq and iq != oq:
        parts.append(iq)
    merged = _normalize(" ".join(parts))
    return merged or _normalize(fallback_query)


def _vector_fallback_manual_chunks(
    conn: sqlite3.Connection,
    parsed: dict,
    query: str,
    top_k: int,
    embed_model: str,
    embed_url: str,
    max_chunks: int,
) -> list[dict]:
    q_text = _compose_vector_query_text(parsed, query)
    q_vec = _embed_text_ollama(q_text, embed_model, embed_url)
    if not q_vec:
        return []

    _ensure_manual_embeddings_table(conn)
    rows = conn.execute(
        """
        SELECT id AS chunk_id, source_file, page_no, chunk_text
        FROM manual_chunks
        LIMIT ?
        """,
        (max_chunks,),
    ).fetchall()

    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        cid = int(row["chunk_id"])
        cvec = _get_or_create_chunk_embedding(
            conn,
            cid,
            str(row["chunk_text"]),
            embed_model,
            embed_url,
        )
        if not cvec or len(cvec) != len(q_vec):
            continue
        sim = _cosine_vectors(q_vec, cvec)
        scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results: list[dict] = []
    for sim, row in scored[: max(1, top_k)]:
        results.append(
            {
                "chunk_id": int(row["chunk_id"]),
                "source_file": row["source_file"],
                "page_no": row["page_no"],
                "snippet": _manual_snippet_for_result(row["chunk_text"]),
                "score": round(float(sim), 4),
                "match_type": "vector",
            }
        )
    conn.commit()
    return results


def _build_ngram_idf(corpus: list[str], n: int = 2) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    total = max(1, len(corpus))
    for text in corpus:
        norm = _compact(text)
        grams = {norm[i : i + n] for i in range(max(0, len(norm) - n + 1))}
        for g in grams:
            if g:
                doc_freq[g] = doc_freq.get(g, 0) + 1
    return {g: math.log((1 + total) / (1 + df)) + 1.0 for g, df in doc_freq.items()}


def _ngram_tfidf_vector(text: str, idf: dict[str, float], n: int = 2) -> dict[str, float]:
    norm = _compact(text)
    if len(norm) < n:
        return {}
    tf: dict[str, float] = {}
    for i in range(len(norm) - n + 1):
        g = norm[i : i + n]
        tf[g] = tf.get(g, 0.0) + 1.0
    return {g: (cnt * idf.get(g, 0.0)) for g, cnt in tf.items() if g in idf}


def _sparse_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, av in a.items():
        dot += av * b.get(k, 0.0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_manual_vocab() -> list[str]:
    global _MANUAL_VOCAB
    if _MANUAL_VOCAB is not None:
        return _MANUAL_VOCAB

    vocab_counter: dict[str, int] = {}
    chunks_path = Path(OUTPUT_DIR) / "manual_chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                for heading_key in ("section_title", "subsection_title"):
                    heading = _normalize(str(row.get(heading_key, "")))
                    if not heading or len(heading) < 2:
                        continue
                    if heading in MANUAL_STOPWORDS:
                        continue
                    vocab_counter[heading] = vocab_counter.get(heading, 0) + 1

    # Keep frequently seen real words first.
    _MANUAL_VOCAB = [k for k, _ in sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)]
    return _MANUAL_VOCAB


def _load_manual_recommend_vocab() -> list[str]:
    global _MANUAL_RECOMMEND_VOCAB
    if _MANUAL_RECOMMEND_VOCAB is not None:
        return _MANUAL_RECOMMEND_VOCAB

    phrase_vocab = _load_manual_vocab()
    token_counter: dict[str, int] = {}
    for phrase in phrase_vocab:
        for raw_token in re.split(r"\s+", _normalize(phrase)):
            token = re.sub(r"[^\w가-힣]", "", raw_token)
            token = _strip_question_suffixes(token)
            if token and len(token) >= 2 and token[-1] in KOREAN_PARTICLE_SUFFIXES:
                token = token[:-1]
            if not _is_valid_manual_term(token):
                continue
            if token in MANUAL_RECOMMEND_STOPWORDS:
                continue
            token_counter[token] = token_counter.get(token, 0) + 1

    # 본문 chunk에서도 단어를 수집해 부품/증상 단어(예: 회전솔, 흡입구)를 포함한다.
    chunks_path = Path(OUTPUT_DIR) / "manual_chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                chunk_text = _normalize(str(row.get("chunk_text", "")))
                if not chunk_text:
                    continue
                for raw_token in re.split(r"[^가-힣]+", chunk_text):
                    token = _normalize(raw_token)
                    if not token:
                        continue
                    if len(token) > 8:
                        continue
                    if token and len(token) >= 2 and token[-1] in KOREAN_PARTICLE_SUFFIXES:
                        token = token[:-1]
                    if not _is_valid_manual_term(token):
                        continue
                    if token in MANUAL_RECOMMEND_STOPWORDS:
                        continue
                    token_counter[token] = token_counter.get(token, 0) + 1

    _MANUAL_RECOMMEND_VOCAB = [
        k for k, _ in sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    ]
    return _MANUAL_RECOMMEND_VOCAB


def _load_manual_sentence_candidates() -> list[str]:
    global _MANUAL_SENTENCE_CANDIDATES
    if _MANUAL_SENTENCE_CANDIDATES is not None:
        return _MANUAL_SENTENCE_CANDIDATES

    candidates: list[str] = []
    chunks_path = Path(OUTPUT_DIR) / "manual_chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as src:
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                chunk_text = _normalize(str(row.get("chunk_text", "")))
                if not chunk_text:
                    continue
                # 문장 단위로 쪼개 추천 후보를 만든다.
                for sent in re.split(r"(?<=[\.\!\?])\s+|•", chunk_text):
                    normalized = _normalize(sent)
                    if len(normalized) < 12 or len(normalized) > 420:
                        continue
                    candidates.append(normalized)
    _MANUAL_SENTENCE_CANDIDATES = list(dict.fromkeys(candidates))
    return _MANUAL_SENTENCE_CANDIDATES


def _recommend_sentences_from_terms(query: str, suggested_terms: list[dict], top_n: int = 3) -> list[dict]:
    query_terms = [
        t
        for t in _extract_terms(query)
        if t not in MANUAL_ANCHOR_STOPWORDS and t not in MANUAL_SENTENCE_QUERY_STOPWORDS
    ]
    term_tokens = [str(item.get("term", "")).strip() for item in suggested_terms]
    term_tokens = [t for t in term_tokens if t]
    signal_terms = list(dict.fromkeys([*query_terms, *term_tokens]))[:8]
    if not signal_terms:
        return []

    candidates = _load_manual_sentence_candidates()
    if not candidates:
        return []

    scored: list[tuple[float, str]] = []
    for sentence in candidates:
        sentence_compact = _compact(sentence)
        matched_query_terms = [
            t for t in query_terms if t in sentence or _compact(t) in sentence_compact
        ]
        matched_reco_terms = [
            t for t in term_tokens if t in sentence or _compact(t) in sentence_compact
        ]
        matched_terms = [t for t in signal_terms if t in sentence or _compact(t) in sentence_compact]
        if not matched_terms or not matched_query_terms:
            continue
        if term_tokens and not matched_reco_terms:
            continue
        if len(query_terms) >= 2 and len(set(matched_query_terms)) < 2:
            continue
        expansion_bonus = sum(_expanded_match_bonus(t, sentence, sentence_compact) for t in signal_terms)
        # 최소 2개 이상 단서가 겹치거나, 형태확장 매칭이 있으면 통과시킨다.
        if len(set([*matched_query_terms, *matched_reco_terms])) < 2 and expansion_bonus <= 0.0:
            continue
        score = float(len(set(matched_terms)))
        # 질문어와 추천어가 함께 포함될수록 점수를 높인다.
        score += 0.7 * len(set(matched_query_terms))
        score += 0.3 * len(set(matched_reco_terms))
        score += expansion_bonus
        scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]
    return [{"query": sentence, "score": round(score, 4)} for score, sentence in top]


def _recommend_terms_from_query(query: str, top_n: int = 6) -> list[dict]:
    global _NGRAM_IDF_CACHE
    normalized = _normalize(query)
    if not normalized:
        return []
    vocab = _load_manual_recommend_vocab()
    if not vocab:
        return []
    query_terms = [t for t in _extract_terms(normalized) if t not in MANUAL_ANCHOR_STOPWORDS]
    seed_terms = list(dict.fromkeys(query_terms))[:3]

    def _pack_terms(scored_terms: list[tuple[str, float]]) -> list[dict]:
        merged: list[dict] = []
        for seed in seed_terms:
            merged.append({"term": seed, "similarity": 1.0})
        for term, score in scored_terms:
            merged.append({"term": term, "similarity": round(score, 4)})
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in merged:
            term = str(item.get("term", "")).strip()
            if not term or term in seen:
                continue
            seen.add(term)
            deduped.append(item)
            if len(deduped) >= top_n:
                break
        return deduped

    def _related_to_query(cand: str) -> bool:
        if not query_terms:
            return True
        return any((qt in cand) or (cand in qt) for qt in query_terms)

    query_compact = _compact(normalized)
    if _NGRAM_IDF_CACHE is None:
        _NGRAM_IDF_CACHE = _build_ngram_idf(vocab, n=2)
    query_sparse = _ngram_tfidf_vector(normalized, _NGRAM_IDF_CACHE, n=2)
    fallback_scored: list[tuple[str, float]] = []
    for cand in vocab[:400]:
        if _compact(cand) == query_compact:
            continue
        cand_sparse = _ngram_tfidf_vector(cand, _NGRAM_IDF_CACHE, n=2)
        sim = _sparse_cosine(query_sparse, cand_sparse)
        if sim > 0.05:
            fallback_scored.append((cand, sim))
    fallback_scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [(term, score) for term, score in fallback_scored if _related_to_query(term)]
    base = filtered or fallback_scored
    return _pack_terms(base)


_SNIPPET_LOCALE_LABELS = {
    "ko": "Korean (한국어)",
    "en": "English",
    "th": "Thai (ภาษาไทย)",
    "vi": "Vietnamese (Tiếng Việt)",
    "id": "Indonesian (Bahasa Indonesia)",
    "ms": "Malay (Bahasa Melayu)",
}

# 번역 프롬프트 메타는 한국어만 쓴다. 영문 지시를 소형 모델이 타깃 언어로 그대로 번역해 섞이는 것을 막는다.
_SNIPPET_LOCALE_PROMPT_KO = {
    "en": "영어",
    "th": "태국어",
    "vi": "베트남어",
    "id": "인도네시아어",
    "ms": "말레이어",
}


def _translation_snippet_extra_instructions() -> str:
    """배포 환경에서만 덧붙이는 번역 보조 지시(코드에 제품·도메인 문구를 넣지 않는다)."""
    env_extra = os.environ.get("TRANSLATION_SNIPPET_INSTRUCTIONS", "").strip()
    path = os.environ.get("TRANSLATION_SNIPPET_INSTRUCTIONS_FILE", "").strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return env_extra
    return env_extra


def _strip_translation_instruction_echo(text: str) -> str:
    """소형 번역 모델이 프롬프트 지시문을 본문으로 출력한 경우 앞부분 제거."""
    s = str(text or "").strip()
    if not s:
        return s
    lines = s.split("\n")
    if len(lines) >= 2:
        head = lines[0].strip()
        lower = head.lower()
        hints = (
            "preserve product",
            "output only",
            "no preamble",
            "translation only",
            "giữ lại tên",
            "chỉ dịch",
            "không có lời",
            "pertahankan nama",
            "เก็บชื่อ",
            "รักษาชื่อผลิตภัณฑ์",
        )
        if any(h in lower or h in head for h in hints):
            return "\n".join(lines[1:]).strip()
    for pat in (
        r"(?is)^\s*Giữ lại tên sản phẩm[^.]*\.\s*",
        r"(?is)^\s*Preserve product names[^.]*\.\s*",
    ):
        s2 = re.sub(pat, "", s)
        if s2 != s:
            return s2.strip()
    return s


# UI에서 지원하는 미리 번역 로케일 (배치 스크립트와 동일하게 유지).
MANUAL_TRANSLATION_LOCALES = ("en", "th", "vi", "id", "ms")


def _ensure_manual_translation_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_chunk_translations (
            chunk_id INTEGER NOT NULL,
            locale TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (chunk_id, locale)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_manual_tr_locale ON manual_chunk_translations(locale)"
    )


def get_cached_manual_translation(chunk_id: int, locale: str) -> str | None:
    """SQLite에 저장된 사전 번역. 없으면 None."""
    loc = str(locale or "").strip().lower()
    if loc == "ko":
        return None
    conn = sqlite3.connect(SQLITE_PATH, timeout=60)
    try:
        _ensure_manual_translation_table(conn)
        row = conn.execute(
            "SELECT translated_text FROM manual_chunk_translations WHERE chunk_id = ? AND locale = ?",
            (int(chunk_id), loc),
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


def bulk_get_cached_manual_translations(
    chunk_ids: list[int],
    locale: str,
    sqlite_path: str = SQLITE_PATH,
) -> dict[int, str]:
    """현재 페이지 등 여러 chunk_id의 사전 번역을 한 번에 조회한다."""
    loc = str(locale or "").strip().lower()
    if loc == "ko":
        return {}
    ids = sorted({int(x) for x in chunk_ids if x is not None})
    if not ids:
        return {}
    conn = sqlite3.connect(sqlite_path, timeout=60)
    try:
        _ensure_manual_translation_table(conn)
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT chunk_id, translated_text FROM manual_chunk_translations
            WHERE locale = ? AND chunk_id IN ({placeholders})
            """,
            (loc, *ids),
        ).fetchall()
        return {int(r[0]): str(r[1]) for r in rows if r and r[1] is not None}
    finally:
        conn.close()


def upsert_manual_translation(
    conn: sqlite3.Connection,
    chunk_id: int,
    locale: str,
    translated_text: str,
) -> None:
    _ensure_manual_translation_table(conn)
    conn.execute(
        """
        INSERT INTO manual_chunk_translations (chunk_id, locale, translated_text, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(chunk_id, locale) DO UPDATE SET
            translated_text = excluded.translated_text,
            updated_at = excluded.updated_at
        """,
        (int(chunk_id), str(locale).strip().lower(), translated_text),
    )


def _manual_translation_live_fallback_enabled() -> bool:
    return os.environ.get("MANUAL_TRANSLATION_LIVE_FALLBACK", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _translate_manual_snippet_live(text_ko: str, target_locale: str) -> str | None:
    """번역 모델로 즉시 번역(미리 저장 없음). 배치 스크립트도 동일 로직 사용."""
    normalized = _normalize(str(text_ko or ""))
    if not normalized or not target_locale or target_locale == "ko":
        return None
    tl = str(target_locale).strip().lower()
    prompt: str | None = None
    if uses_translategemma_translation():
        src = TRANSLATION_LOCALE_ISO.get("ko")
        tgt = TRANSLATION_LOCALE_ISO.get(tl)
        if src and tgt:
            prompt = build_translategemma_prompt(src[0], src[1], tgt[0], tgt[1], normalized)
    if prompt is None:
        lang_ko = _SNIPPET_LOCALE_PROMPT_KO.get(
            tl, _SNIPPET_LOCALE_LABELS.get(tl, tl)
        )
        body = (
            f"아래 한국어 텍스트를 {lang_ko}로 번역한다.\n"
            "원문의 의미·절차·경고를 바꾸지 않는다. 고유명사·기술 용어는 가능하면 유지한다.\n"
            "번역문만 출력한다. 머리말이나 부연 없음.\n"
        )
        extra = _translation_snippet_extra_instructions()
        if extra:
            body = f"{body}\n{extra}\n"
        prompt = f"{body}\n{normalized}"
    mchars = _manual_translation_max_chars()
    mtok = _manual_translation_max_new_tokens()
    out = generate_text(
        prompt,
        max_new_tokens=mtok,
        temperature=0.0,
        model=get_translation_model(),
    )
    if not out:
        return None
    stripped = _strip_translation_instruction_echo(out.strip())
    cleaned = _soft_truncate_translation(stripped, mchars)
    return cleaned or None


def translate_manual_snippet(
    text_ko: str,
    target_locale: str,
    chunk_id: int | None = None,
) -> str | None:
    """매뉴얼 스니펫 번역: SQLite 사전 번역 우선, 없으면(설정 시) 실시간 번역."""
    if not str(text_ko or "").strip() or not target_locale or str(target_locale).strip().lower() == "ko":
        return None
    loc = str(target_locale).strip().lower()
    if chunk_id is not None:
        cached = get_cached_manual_translation(int(chunk_id), loc)
        if cached:
            return cached
    if _manual_translation_live_fallback_enabled():
        return _translate_manual_snippet_live(text_ko, loc)
    return None


def _split_chunks(text: str, size: int = 380, overlap: int = 80) -> list[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Keep bullet points separable so "기타" pages don't become one giant chunk.
    raw = re.sub(r"\s*•\s*", "\n• ", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if not lines:
        return []

    units: list[str] = []
    for line in lines:
        parts = [p.strip() for p in re.split(r"(?<=\.)\s+", line) if p.strip()]
        units.extend(parts if parts else [line])

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:].strip() if overlap > 0 else ""
            current = f"{tail} {unit}".strip() if tail else unit
        else:
            chunks.append(unit[:size].strip())
            current = unit[size - overlap :].strip() if overlap > 0 else ""
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _ensure_manual_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_manual_page ON manual_chunks(page_no)")


def _build_index(conn: sqlite3.Connection, pdf_path: Path) -> bool:
    PdfReader = None
    try:
        from pypdf import PdfReader as _PdfReader

        PdfReader = _PdfReader
    except ModuleNotFoundError:
        try:
            from PyPDF2 import PdfReader as _PdfReader

            PdfReader = _PdfReader
        except ModuleNotFoundError:
            PdfReader = None

    _ensure_manual_table(conn)
    existing = conn.execute("SELECT COUNT(1) FROM manual_chunks").fetchone()[0]
    if existing > 0:
        return True

    rows: list[tuple[str, int, str]] = []

    if PdfReader is not None and pdf_path.exists():
        reader = PdfReader(str(pdf_path))
        for idx, page in enumerate(reader.pages, start=1):
            page_text = _normalize(page.extract_text() or "")
            if not page_text:
                continue
            for chunk in _split_chunks(page_text):
                rows.append((pdf_path.name, idx, chunk))
    else:
        chunks_path = Path(OUTPUT_DIR) / "manual_chunks.jsonl"
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    chunk_text = _normalize(str(row.get("chunk_text", "")))
                    if not chunk_text:
                        continue
                    page_no = row.get("page_start") or row.get("page_no")
                    if not page_no:
                        page_nos = row.get("page_nos") or []
                        page_no = page_nos[0] if page_nos else 1
                    source_file = str(row.get("source_file") or pdf_path.name)
                    rows.append((source_file, int(page_no), chunk_text))

    if rows:
        conn.executemany(
            "INSERT INTO manual_chunks (source_file, page_no, chunk_text) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        return True
    return False


def ensure_manual_index(sqlite_path: str = SQLITE_PATH) -> bool:
    global _INDEX_READY
    conn = sqlite3.connect(sqlite_path, timeout=60)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_manual_table(conn)
        _ensure_manual_translation_table(conn)
        conn.commit()
        if _INDEX_READY:
            return True
        _INDEX_READY = _build_index(conn, MANUAL_PDF_PATH)
        return _INDEX_READY
    finally:
        conn.close()


def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", str(text or "")))


def _intent_matches_question_language(user_query: str, intent: str) -> bool:
    """한글 질문인데 intent만 라틴 문자면 버린다 (모델이 영어로만 요약하는 경우)."""
    if _contains_hangul(user_query) and intent.strip():
        return _contains_hangul(intent)
    return True


def _intent_fallback_terms(user_query: str) -> str:
    """토큰만으로 intent_query 문자열 구성 (LLM 없음)."""
    terms = _extract_terms(user_query)
    return _normalize(" ".join(terms) if terms else user_query)


def _llm_intent_query_llm_only(user_query: str) -> str:
    """2단계용: 메인 LLM으로 intent_query JSON 정제. 실패 시 토큰 fallback."""
    fallback = _intent_fallback_terms(user_query)
    lang_rule = (
        "- intent_query 언어는 사용자 질문과 같게: 한국어 질문이면 **한국어 검색어**만 사용한다. "
        "영어 번역·영문 요약을 넣지 마라.\n"
        if _contains_hangul(user_query)
        else "- intent_query는 사용자 질문과 같은 언어로 검색에 맞게만 다듬는다.\n"
    )
    prompt = (
        "사용자 매뉴얼 질문을 검색용 짧은 문장으로 정리해 JSON으로 출력하라. 설명 금지.\n"
        f"{lang_rule}"
        '출력 형식: {"intent_query":"..."}\n'
        f"질문: {user_query}"
    )
    refined = generate_text(
        prompt,
        max_new_tokens=96,
        temperature=0.0,
        model=get_manual_intent_model(),
    )
    if not refined:
        return fallback
    text = _normalize(refined)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start <= end:
        try:
            obj = json.loads(text[start : end + 1])
            intent = _normalize(obj.get("intent_query", ""))
        except Exception:  # noqa: BLE001
            intent = text
    else:
        intent = text

    if not intent or len(intent) > 80:
        return fallback
    if not _intent_matches_question_language(user_query, intent):
        return fallback
    generic = {"질문", "사용자", "매뉴얼", "찾아", "알려", "방법"}
    if sum(1 for t in generic if t in intent) >= 3:
        return fallback
    return intent


def _llm_intent_query(user_query: str) -> str:
    # parse_manual_query 등 단일 경로용: MANUAL_INTENT_USE_LLM=1 일 때만 LLM, 아니면 토큰만.
    use_llm = os.environ.get("MANUAL_INTENT_USE_LLM", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not use_llm:
        return _intent_fallback_terms(user_query)
    return _llm_intent_query_llm_only(user_query)


def _manual_parsed_bundle(
    user_query: str,
    intent_query: str,
    *,
    include_suggested_terms: bool = False,
) -> dict:
    raw_terms = _extract_terms(user_query)
    intent_terms = _extract_terms(intent_query)
    search_terms = list(dict.fromkeys([*raw_terms, *intent_terms]))
    anchor_terms = _select_anchor_terms(raw_terms=raw_terms, all_terms=search_terms)
    suggested_terms = _recommend_terms_from_query(user_query) if include_suggested_terms else []
    suggested_queries = (
        _recommend_sentences_from_terms(user_query, suggested_terms) if include_suggested_terms else []
    )
    return {
        "original_query": _normalize(user_query),
        "intent_query": intent_query,
        "raw_terms": raw_terms,
        "intent_terms": intent_terms,
        "search_terms": search_terms,
        "anchor_terms": anchor_terms,
        "suggested_terms": suggested_terms,
        "suggested_queries": suggested_queries,
    }


def parse_manual_query_for_ui(user_query: str) -> dict:
    """LLM intent 없이 토큰 분리 결과만 채운다 (재고 검색 성공 시 UI용)."""
    raw_terms = _extract_terms(user_query)
    search_terms = list(dict.fromkeys(raw_terms))
    anchor_terms = _select_anchor_terms(raw_terms=raw_terms, all_terms=search_terms)
    return {
        "original_query": _normalize(user_query),
        "intent_query": "",
        "raw_terms": raw_terms,
        "intent_terms": [],
        "search_terms": search_terms,
        "anchor_terms": anchor_terms,
        "suggested_terms": [],
        "suggested_queries": [],
    }


def parse_manual_query(user_query: str, include_suggested_terms: bool = False) -> dict:
    intent_query = _llm_intent_query(user_query)
    return _manual_parsed_bundle(
        user_query,
        intent_query,
        include_suggested_terms=include_suggested_terms,
    )


def _manual_vector_fallback_enabled() -> bool:
    return os.environ.get("MANUAL_VECTOR_FALLBACK", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _keyword_sql_manual_chunks(
    conn: sqlite3.Connection,
    parsed: dict,
    top_k: int,
    *,
    stage: int,
) -> list[dict]:
    """키워드 OR SQL + 점수. 임계값 미만이면 빈 리스트. match_type: keyword_stage1 / keyword_stage2."""
    _ = top_k
    terms = parsed.get("search_terms") or []
    must_terms = parsed.get("anchor_terms") or []
    if not terms:
        return []

    where_parts: list[str] = []
    params: list[str] = []
    for term in terms:
        compact_term = _compact(term)
        where_parts.append(
            "(chunk_text LIKE ? OR REPLACE(REPLACE(chunk_text, ' ', ''), char(10), '') LIKE ?)"
        )
        params.extend([f"%{term}%", f"%{compact_term}%"])
    where = " OR ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT id AS chunk_id, source_file, page_no, chunk_text
        FROM manual_chunks
        WHERE {where}
        LIMIT 1000
        """,
        params,
    ).fetchall()

    def _score_rows(require_anchor_terms: bool) -> list[tuple[float, sqlite3.Row]]:
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            raw_text = str(row["chunk_text"])
            text_norm = _normalize(raw_text)
            text_compact = _compact(raw_text)
            if require_anchor_terms and must_terms and not any(
                _contains_term(t, text_norm=text_norm, text_compact=text_compact)
                for t in must_terms
            ):
                continue
            matched = sum(
                1
                for t in terms
                if _contains_term(t, text_norm=text_norm, text_compact=text_compact)
            )
            score = float(matched)
            score += sum(_expanded_match_bonus(t, text_norm, text_compact) for t in terms)
            score -= _boilerplate_penalty(text_norm)
            scored.append((score, row))
        return scored

    reranked: list[tuple[float, sqlite3.Row]] = _score_rows(require_anchor_terms=True)
    if not reranked:
        reranked = _score_rows(require_anchor_terms=False)

    reranked.sort(key=lambda x: x[0], reverse=True)
    threshold = 2.0
    tag = f"keyword_stage{stage}"
    results: list[dict] = []
    for score, row in reranked:
        if score < threshold:
            continue
        chunk_id = int(row["chunk_id"])
        snippet_ko = _manual_snippet_for_result(row["chunk_text"])
        results.append(
            {
                "chunk_id": chunk_id,
                "source_file": row["source_file"],
                "page_no": row["page_no"],
                "snippet": snippet_ko,
                "score": round(score, 4),
                "match_type": tag,
            }
        )
    return _dedupe_similar_manual_hits(results)


def search_manual_chunks(
    query: str,
    sqlite_path: str = SQLITE_PATH,
    top_k: int = 5,
    manual_parsed: dict | None = None,
) -> list[dict]:
    """1) 토큰만 SQL → 2) LLM 의도 후 SQL → 3) 벡터. manual_parsed는 호환용으로 무시한다."""
    query = query.strip()
    if not query:
        return []
    _ = manual_parsed

    global _INDEX_READY
    conn = sqlite3.connect(sqlite_path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        _ensure_manual_table(conn)
        if not _INDEX_READY:
            _INDEX_READY = _build_index(conn, MANUAL_PDF_PATH)

        try:
            mc = int(os.environ.get("MANUAL_VECTOR_MAX_CHUNKS", "5000"))
        except ValueError:
            mc = 5000

        # 1단계: 토큰만으로 intent 구성 → SQLite 키워드 검색
        parsed1 = _manual_parsed_bundle(query, _intent_fallback_terms(query))
        hits = _keyword_sql_manual_chunks(conn, parsed1, top_k, stage=1)
        if hits:
            return hits

        # 2단계: LLM 의도 정제 → 동일 SQL 검색
        parsed2 = _manual_parsed_bundle(query, _llm_intent_query_llm_only(query))
        hits = _keyword_sql_manual_chunks(conn, parsed2, top_k, stage=2)
        if hits:
            return hits

        # 3단계: 임베딩 유사도 (2단계 파싱 기준으로 질의 텍스트 구성)
        if _manual_vector_fallback_enabled():
            vec = _vector_fallback_manual_chunks(
                conn,
                parsed2,
                query,
                top_k,
                DEFAULT_EMBED_MODEL,
                DEFAULT_EMBED_URL,
                max_chunks=max(1, mc),
            )
            return _dedupe_similar_manual_hits(vec)
        return []
    finally:
        conn.close()


def build_manual_answer(
    query: str,
    manual_hits: list[dict],
    max_refs: int = 3,
    response_locale: str = "ko",
) -> str:
    hits = manual_hits[: max(1, int(max_refs))]
    if not hits:
        return "매뉴얼에서 직접 확인된 내용이 없습니다."
    refs = ", ".join(f"p.{item.get('page_no', '?')}" for item in hits)
    template = MANUAL_ANSWER_TEMPLATES.get(response_locale, MANUAL_ANSWER_TEMPLATES["ko"])
    return template.format(query=query, refs=refs)
