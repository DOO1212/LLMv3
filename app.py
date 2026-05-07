"""재고 검색용 Streamlit 앱 UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from io import BytesIO
import json

from config import SQLITE_PATH
from llm_fallback import warmup_local_model
from router import route_query


SEARCH_LIMIT = 100000
LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
}


def _result_rows(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(results, start=1):
        record = item.get("record", {})
        structured = record.get("structured_data", {})
        raw = record.get("raw_data", {})
        rows.append(
            {
                "No": idx,
                "재고ID": raw.get("재고ID", ""),
                "상품코드": raw.get("상품코드", "") or structured.get("product_code", ""),
                "품목명": structured.get("product_name", ""),
                "카테고리": structured.get("category", ""),
                "창고": structured.get("warehouse", ""),
                "재고": structured.get("stock", ""),
                "단가": structured.get("price", ""),
                "공급업체": structured.get("supplier", ""),
                "상태": structured.get("status", ""),
            }
        )
    return rows


def _to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


st.set_page_config(page_title="챗봇", layout="wide")
warmup_local_model()

st.markdown(
    """
    <style>
    .block-container { max-width: 80%; padding-top: 3rem; }
    .stDataFrameToolbar { display: none !important; }
    div.stFormSubmitButton > button {
        height: 3rem;
        margin-top: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "question_locale" not in st.session_state:
    st.session_state["question_locale"] = "ko"
if "locale" not in st.session_state:
    st.session_state["locale"] = "ko"

with st.form("search_form", clear_on_submit=False, border=False):
    main_left, main_right = st.columns([8, 2])
    with main_left:
        st.markdown("카테고리 예시를 참고해 자유롭게 질문해 주세요.")
        st.caption("생활용품, 식품, 의류, 전자, 부품, 사무용품, 금형, PCB, 사출, 포장")
        query = st.text_input(
            "query",
            placeholder="예: 식품 카테고리 전체 품목 알려줘",
            label_visibility="collapsed",
        )

    with main_right:
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        st.selectbox(
            "질문 언어",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key="question_locale",
            label_visibility="visible",
        )
        st.selectbox(
            "답변 언어",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key="locale",
            label_visibility="visible",
        )
        st.markdown("<div style='margin-top:0.35rem'></div>", unsafe_allow_html=True)
        search_clicked = st.form_submit_button("검색", use_container_width=True)

if search_clicked:
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        routed = route_query(
            query.strip(),
            sqlite_path=SQLITE_PATH,
            top_k=SEARCH_LIMIT,
            response_locale=st.session_state["locale"],
        )
        st.session_state["last_query"] = query.strip()
        st.session_state["last_routed"] = routed


if "last_routed" in st.session_state:
    routed = st.session_state["last_routed"]
    route = routed.get("route")
    st.caption(f"route: {route}")
    with st.expander("파싱 결과 보기"):
        parsed_query = routed.get("parsed_query") or {}
        st.code(
            json.dumps(parsed_query, ensure_ascii=False, indent=2),
            language="json",
        )

    if route == "llm_fallback":
        st.info(routed.get("answer") or "답변이 없습니다.")

    else:
        results = routed.get("results") or []
        if not results:
            st.info("검색 결과가 없습니다.")
        else:
            result_df = pd.DataFrame(_result_rows(results))
            left, right = st.columns([6, 2])
            with left:
                st.caption(f"총 {len(result_df)}건")
            with right:
                st.download_button(
                    "검색결과 다운로드 (.xlsx)",
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
