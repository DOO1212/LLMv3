"""Local Llama fallback answers for queries SQLite search cannot satisfy."""

from __future__ import annotations

import json
import os
from threading import Lock
import urllib.error
import urllib.request

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from i18n import t as t_ui


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_LLAMA_MODEL = "llama3.3:70b"
DEFAULT_OLLAMA_TIMEOUT = 240
DEFAULT_LLM_BACKEND = "ollama"

_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None
_LOCAL_LOAD_LOCK = Lock()


def get_ollama_timeout():
    try:
        return int(os.environ.get("OLLAMA_TIMEOUT", DEFAULT_OLLAMA_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_OLLAMA_TIMEOUT


def _get_llm_backend():
    return os.environ.get("LLM_BACKEND", DEFAULT_LLM_BACKEND).strip().lower()


def _load_local_model():
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR
    with _LOCAL_LOAD_LOCK:
        if _LOCAL_MODEL is None or _LOCAL_PROCESSOR is None:
            model_id = os.environ.get("LLAMA_MODEL", DEFAULT_LLAMA_MODEL)
            _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            _LOCAL_MODEL = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    return _LOCAL_MODEL, _LOCAL_PROCESSOR


def warmup_local_model() -> None:
    """앱 시작 시 로컬 모델을 안전하게 1회 로딩한다."""
    if _get_llm_backend() != "local_transformers":
        return
    _load_local_model()


def generate_text(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str | None:
    backend = _get_llm_backend()
    if backend == "ollama":
        ollama_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        model = os.environ.get("LLAMA_MODEL", DEFAULT_LLAMA_MODEL)
        timeout = get_ollama_timeout()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        request = urllib.request.Request(
            ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError:
            return None
        return (response_data.get("response") or "").strip() or None

    model, processor = _load_local_model()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        chat_template_kwargs={"enable_thinking": False},
    ).to(model.device)
    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 0.01) if do_sample else 1.0,
    )
    text = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return text or None


LOCALE_RESPONSE_LANG = {
    "ko": "한국어(Korean)",
    "en": "English",
    "th": "ภาษาไทย(Thai)",
    "vi": "Tiếng Việt(Vietnamese)",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu(Malay)",
}


def build_prompt(query, parsed_query=None, search_error=None, locale="ko"):
    parsed_query = parsed_query or {}
    search_error_text = str(search_error) if search_error else "없음"
    lang = LOCALE_RESPONSE_LANG.get(locale, LOCALE_RESPONSE_LANG["ko"])

    return f"""
너는 재고 검색 앱의 로컬 Llama fallback assistant다.
SQLite 검색에서 결과가 없거나 파싱이 실패했을 때만 호출된다.
실제 DB 검색 결과를 알 수 없으므로 특정 상품이 있다고 단정하지 마라.
사용자의 질문을 어떻게 해석했는지 설명하고, 검색 가능한 질문 형태를 짧게 제안하라.
**반드시 아래 언어로만 답하라: {lang}.**

사용자 질문:
{query}

파싱 결과:
{json.dumps(parsed_query, ensure_ascii=False)}

검색 오류:
{search_error_text}

fallback 답변:
""".strip()


def answer_with_llm(query, parsed_query=None, search_error=None, locale="ko"):
    answer = generate_text(
        build_prompt(query, parsed_query, search_error, locale=locale),
        max_new_tokens=384,
        temperature=0.7,
    )
    if answer:
        return answer

    return t_ui(locale, "llm_offline")
