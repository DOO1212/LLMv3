"""재고 검색용 Streamlit 앱 UI."""

from __future__ import annotations

import html
import re
import pandas as pd
import streamlit as st
from io import BytesIO
import json

from config import SQLITE_PATH
from llm_fallback import (
    TRANSLATION_LOCALE_ISO,
    build_translategemma_prompt,
    generate_text,
    get_translation_model,
    uses_translategemma_translation,
    warmup_local_model,
)
from manual_search import bulk_get_cached_manual_translations, translate_manual_snippet
from router import route_query


SEARCH_LIMIT = 100000
MANUAL_PAGE_SIZE = 5
LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
}


def _translate_query_for_search(query_text: str, question_locale: str) -> str:
    source = str(question_locale or "ko").strip().lower()
    normalized = query_text.strip()
    if not normalized or source == "ko":
        return normalized
    model_tr = get_translation_model()
    if uses_translategemma_translation():
        pair_src = TRANSLATION_LOCALE_ISO.get(source)
        pair_ko = TRANSLATION_LOCALE_ISO.get("ko")
        if pair_src and pair_ko:
            prompt = build_translategemma_prompt(
                pair_src[0], pair_src[1], pair_ko[0], pair_ko[1], normalized
            )
            translated = generate_text(
                prompt,
                max_new_tokens=160,
                temperature=0.0,
                model=model_tr,
            )
            if translated:
                return translated.strip() or normalized
            return normalized
    prompt = (
        "Translate the user query into Korean for manual/product search.\n"
        "Return JSON only: {\"ko_query\":\"...\"}\n"
        "Rules:\n"
        "- Keep product/part names as-is when they are proper nouns.\n"
        "- Keep it concise and search-friendly.\n"
        f"Source locale: {source}\n"
        f"User query: {normalized}"
    )
    translated = generate_text(
        prompt,
        max_new_tokens=96,
        temperature=0.0,
        model=model_tr,
    )
    if not translated:
        return normalized
    text = translated.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        return normalized
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return normalized
    ko_query = str(payload.get("ko_query", "")).strip()
    return ko_query or normalized


def _run_search(query_text: str) -> None:
    question_locale = st.session_state.get("question_locale", "ko")
    search_query = _translate_query_for_search(query_text, question_locale)
    routed = route_query(
        query_text.strip(),
        sqlite_path=SQLITE_PATH,
        top_k=SEARCH_LIMIT,
        response_locale=st.session_state["locale"],
        search_query=search_query,
    )
    st.session_state["last_query"] = query_text.strip()
    st.session_state["last_routed"] = routed
    st.session_state["manual_page"] = 1
    st.session_state["manual_snippet_tr_cache"] = {}


def _result_rows(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(results, start=1):
        record = item.get("record", {})
        structured = record.get("structured_data", {})
        raw = record.get("raw_data", {})
        stock = structured.get("stock")
        safety_stock = structured.get("safety_stock")
        try:
            stock_num = int(stock) if stock is not None else None
        except (TypeError, ValueError):
            stock_num = None
        try:
            safety_num = int(safety_stock) if safety_stock is not None else None
        except (TypeError, ValueError):
            safety_num = None
        shortage = (
            max(safety_num - stock_num, 0)
            if stock_num is not None and safety_num is not None
            else ""
        )
        rows.append(
            {
                "No": idx,
                "재고ID": raw.get("재고ID", ""),
                "상품코드": raw.get("상품코드", "") or structured.get("product_code", ""),
                "품목명": structured.get("product_name", ""),
                "카테고리": structured.get("category", ""),
                "창고": structured.get("warehouse", ""),
                "재고": structured.get("stock", ""),
                "안전재고": structured.get("safety_stock", ""),
                "부족수량": shortage,
                "단가": structured.get("price", ""),
                "공급업체": structured.get("supplier", ""),
                "상태": structured.get("status", ""),
            }
        )
    return rows


def _to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    # openpyxl cannot write ASCII control chars except tab/newline/carriage return.
    illegal_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == object:
            safe_df[col] = safe_df[col].map(
                lambda v: illegal_chars.sub("", v) if isinstance(v, str) else v
            )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        safe_df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


def _manual_rows(manual_results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(manual_results, start=1):
        rows.append(
            {
                "No": idx,
                "문서": item.get("source_file", ""),
                "페이지": item.get("page_no", ""),
                "내용": item.get("snippet", ""),
            }
        )
    return rows


st.set_page_config(page_title="재고 · 매뉴얼 검색", page_icon="📦", layout="wide")
warmup_local_model()

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Noto+Sans+KR:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    :root {
      --ink: #0f172a;
      --ink-muted: #475569;
      --surface: #ffffff;
      --surface-soft: #f8fafc;
      --line: rgba(15, 23, 42, 0.08);
      --accent: #0d9488;
      --accent-hover: #0f766e;
      --radius-lg: 18px;
      --shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
    }
    .stApp {
      font-family: "Noto Sans KR", "DM Sans", system-ui, sans-serif;
      background: #ffffff !important;
    }
    .main {
      background-color: #ffffff !important;
    }
    header[data-testid="stHeader"] {
      padding-top: 0.35rem !important;
      padding-bottom: 0.35rem !important;
    }
    .main .block-container,
    section[data-testid="stMain"] .block-container {
      max-width: 960px;
      margin-left: auto;
      margin-right: auto;
      padding: 1.75rem 1.25rem 3rem;
    }
    h1.app-hero-title {
      font-family: "DM Sans", "Noto Sans KR", sans-serif;
      font-weight: 700;
      font-size: clamp(1.65rem, 2.6vw, 2.1rem);
      letter-spacing: -0.03em;
      color: var(--ink);
      margin: 0 0 0.35rem 0;
      line-height: 1.2;
    }
    div[data-testid="stForm"] {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      padding: 1.35rem 1.5rem 1.5rem;
    }
    div[data-testid="stForm"] label[data-testid="stWidgetLabel"] p {
      font-weight: 600;
      color: var(--ink);
      font-size: 0.88rem;
    }
    div[data-testid="stForm"] .stTextInput input {
      border-radius: 12px !important;
      border: 1px solid var(--line) !important;
      padding: 0.75rem 0.95rem !important;
      font-size: 1rem !important;
    }
    div[data-testid="stForm"] .stTextInput input:focus {
      border-color: rgba(13, 148, 136, 0.45) !important;
      box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12) !important;
    }
    div[data-testid="stForm"] div[data-baseweb="select"] > div {
      border-radius: 10px !important;
    }
    .stDataFrameToolbar { display: none !important; }
    div[data-testid="stExpander"] {
      border: 1px solid var(--line) !important;
      border-radius: 12px !important;
      background: var(--surface-soft) !important;
    }
    .route-pill {
      display: inline-block;
      font-family: "DM Sans", monospace;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      padding: 0.28rem 0.65rem;
      border-radius: 999px;
      background: rgba(13, 148, 136, 0.12);
      color: #0f766e;
      border: 1px solid rgba(13, 148, 136, 0.25);
    }
    .manual-hit-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1rem 1.15rem 1.1rem;
      background: var(--surface);
      margin-bottom: 0.85rem;
      box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
    }
    .manual-hit-meta {
      font-size: 0.92rem;
      font-weight: 600;
      color: var(--ink);
      margin-bottom: 0.35rem;
    }
    .manual-hit-score {
      font-size: 0.8rem;
      color: var(--ink-muted);
      font-family: "DM Sans", monospace;
      margin-bottom: 0.65rem;
    }
    div[data-testid="column"] .stButton button {
      border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "question_locale" not in st.session_state:
    st.session_state["question_locale"] = "ko"
if "locale" not in st.session_state:
    st.session_state["locale"] = "ko"
if "manual_page" not in st.session_state:
    st.session_state["manual_page"] = 1

st.markdown(
    """
    <div style="padding:0.15rem 0 0.65rem;">
      <h1 class="app-hero-title">재고 · 매뉴얼 검색</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("search_form", clear_on_submit=False, border=False):
    st.markdown("**검색**")
    st.caption("생활용품, 식품, 의류, 전자, 부품, 사무용품, 금형, PCB, 사출, 포장")
    lang_left, lang_right = st.columns(2, gap="medium")
    with lang_left:
        st.selectbox(
            "질문 언어",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key="question_locale",
        )
    with lang_right:
        st.selectbox(
            "답변 언어",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key="locale",
        )
    query = st.text_input(
        "검색어",
        placeholder="예: 식품 카테고리 전체 품목 알려줘",
        label_visibility="collapsed",
    )
    search_clicked = st.form_submit_button("검색하기", use_container_width=True)

if search_clicked:
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        _run_search(query.strip())


if "last_routed" in st.session_state:
    routed = st.session_state["last_routed"]
    route = routed.get("route")
    parsed_query = routed.get("parsed_query") or {}

    st.divider()
    head_left, head_right = st.columns([3, 1.2], vertical_alignment="center")
    with head_left:
        st.markdown("### 결과")
    with head_right:
        route_label = html.escape(str(route or "-"))
        st.markdown(
            f'<p style="text-align:right;margin:0;"><span class="route-pill">{route_label}</span></p>',
            unsafe_allow_html=True,
        )

    with st.expander("파싱·라우팅 JSON 보기", expanded=False):
        st.code(json.dumps(parsed_query, ensure_ascii=False, indent=2), language="json")

    if route == "manual_pdf":
        manual_results = routed.get("manual_results") or []
        if not manual_results:
            st.info("검색 결과가 없습니다.")
        else:
            manual_answer = routed.get("answer")
            if manual_answer:
                st.success(manual_answer)
            st.caption(f"매뉴얼 {len(manual_results)}건 · 페이지당 {MANUAL_PAGE_SIZE}건")
            total_pages = max(1, (len(manual_results) + MANUAL_PAGE_SIZE - 1) // MANUAL_PAGE_SIZE)
            current_page = max(1, min(st.session_state.get("manual_page", 1), total_pages))
            st.session_state["manual_page"] = current_page
            pager_left, pager_mid, pager_right = st.columns([1, 2, 1])
            with pager_left:
                if st.button("이전", use_container_width=True, disabled=current_page <= 1):
                    st.session_state["manual_page"] = current_page - 1
                    st.rerun()
            with pager_mid:
                st.markdown(
                    f"<p style='text-align:center;margin:0.35rem 0;color:#64748b;font-size:0.9rem;'>"
                    f"{current_page} / {total_pages}</p>",
                    unsafe_allow_html=True,
                )
            with pager_right:
                if st.button(
                    "다음",
                    use_container_width=True,
                    disabled=current_page >= total_pages,
                ):
                    st.session_state["manual_page"] = current_page + 1
                    st.rerun()

            start = (current_page - 1) * MANUAL_PAGE_SIZE
            end = start + MANUAL_PAGE_SIZE
            page_items = manual_results[start:end]
            tr_cache = st.session_state.setdefault("manual_snippet_tr_cache", {})
            locale = st.session_state.get("locale", "ko")
            if locale != "ko":
                chunk_ids_page = [
                    item.get("chunk_id") for item in page_items if item.get("chunk_id") is not None
                ]
                for cid, text in bulk_get_cached_manual_translations(chunk_ids_page, locale).items():
                    tr_cache[f"{locale}:{cid}"] = text
            for idx, item in enumerate(page_items, start=start + 1):
                source_file = item.get("source_file", "")
                page_no = item.get("page_no", "")
                locale = st.session_state.get("locale", "ko")
                base_snippet = item.get("snippet", "")
                if locale != "ko":
                    ck = f"{locale}:{item.get('chunk_id')}"
                    if ck not in tr_cache:
                        tr_cache[ck] = (
                            translate_manual_snippet(
                                base_snippet,
                                locale,
                                item.get("chunk_id"),
                            )
                            or base_snippet
                        )
                    snippet = tr_cache[ck]
                else:
                    snippet = base_snippet
                score = item.get("score")
                score_text = f"{float(score):.4f}" if score is not None else "-"
                meta = html.escape(f"{idx}. {source_file} / p.{page_no}")
                body = html.escape(snippet).replace("\n", "<br/>")
                st.markdown(
                    f'<div class="manual-hit-card">'
                    f'<div class="manual-hit-meta">{meta}</div>'
                    f'<div class="manual-hit-score">score · {html.escape(score_text)}</div>'
                    f'<div style="font-size:0.95rem;line-height:1.55;color:#1e293b;">{body}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        results = routed.get("results") or []
        if not results:
            st.info("검색 결과가 없습니다.")
        else:
            result_df = pd.DataFrame(_result_rows(results))
            bar_left, bar_right = st.columns([4, 2], vertical_alignment="center")
            with bar_left:
                st.markdown(
                    f"<p style='margin:0;color:#64748b;font-size:0.92rem;'>총 <strong>{len(result_df)}</strong>건</p>",
                    unsafe_allow_html=True,
                )
            with bar_right:
                st.download_button(
                    "엑셀 다운로드",
                    data=_to_xlsx_bytes(result_df),
                    file_name="search_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )
