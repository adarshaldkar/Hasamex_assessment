from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Iterable

import streamlit as st

from src.config import get_settings
from src.core.chunker import chunk_transcripts
from src.core.embedder import HashEmbeddingProvider, SentenceTransformerEmbeddingProvider
from src.core.extractor import (
    GeminiExtractionProvider,
    MockExtractionProvider,
    OpenAIExtractionProvider,
    build_ground_truth_matrix,
    load_ground_truth_matrix,
)
from src.core.parser import compute_sha256, parse_transcript_file, parse_transcript_text
from src.core.quote_verifier import resolve_evidence_turn_ids
from src.core.rag_engine import GeminiRAGProvider, MockRAGProvider, OpenAIRAGProvider, RAGEngine
from src.core.retriever import HybridRetriever, InMemoryVectorIndex
from src.core.synthesizer import (
    GeminiSynthesisProvider,
    MockSynthesisProvider,
    OpenAISynthesisProvider,
    build_synthesis_report,
    load_synthesis_report,
    synthesis_cache_is_valid,
)
from src.models.analysis import GroundTruthMatrix, QuoteCitation, TranscriptAnalysis
from src.models.retrieval import RAGResponse
from src.models.synthesis import SynthesisReport
from src.models.transcript import TranscriptDocument

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = get_settings()
RAW_DIR = (ROOT / "data" / "raw") if (ROOT / "data" / "raw").exists() else (ROOT / "data")
UPLOAD_DIR = RAW_DIR / "uploads"
PROCESSED_DIR = ROOT / "storage" / "processed"
MATRIX_PATH = PROCESSED_DIR / "ground_truth_matrix.json"
SYNTHESIS_PATH = PROCESSED_DIR / "synthesis_report.json"
GUIDE_PATH = RAW_DIR / "Interview_Guide.txt"

DEFAULT_TRANSCRIPT_SPECS = [
    ("Transcript_1_France.txt", "france_001"),
    ("Transcript_2_Germany.txt", "germany_001"),
    ("Transcript_3_UK.txt", "uk_001"),
]

STATUS_STYLES = {
    "EXACT_VERIFIED": ("#10b981", "✓ EXACT_VERIFIED"),
    "NORMALIZED_VERIFIED": ("#0ea5e9", "≈ NORMALIZED"),
    "FALLBACK_EVIDENCE": ("#f59e0b", "⚠ FALLBACK"),
    "REJECTED": ("#ef4444", "✕ REJECTED"),
}

st.set_page_config(
    page_title="Transcript Intelligence Pro",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0b1220;
          --surface: #111827;
          --surface-2: #182235;
          --border: #25324a;
          --text: #f8fafc;
          --muted: #94a3b8;
          --cyan: #22d3ee;
          --emerald: #10b981;
        }
        .stApp { background: var(--bg); color: var(--text); }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
        .hero {
          padding: 1.15rem 1.35rem;
          border: 1px solid var(--border);
          border-radius: 16px;
          background: linear-gradient(135deg, #101827 0%, #0f172a 55%, #0b2330 100%);
          margin-bottom: 1rem;
        }
        .hero-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }
        .hero-sub { color: var(--muted); margin-top: .25rem; }
        .metric-card, .panel-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 1rem;
          height: 100%;
        }
        .metric-value { font-size: 1.45rem; font-weight: 750; }
        .metric-label { color: var(--muted); font-size: .83rem; }
        .pill {
          display: inline-block;
          padding: .18rem .48rem;
          border-radius: 999px;
          margin: .1rem .2rem .1rem 0;
          font-size: .78rem;
          background: #162134;
          color: #dbeafe;
          border: 1px solid #2c3c57;
        }
        .quote-box {
          background: #0b1322;
          border-left: 3px solid var(--cyan);
          border-radius: 8px;
          padding: .75rem .9rem;
          color: #e2e8f0;
          line-height: 1.55;
        }
        .answer-box { line-height: 1.55; color: #e5e7eb; }
        .timestamp { font-variant-numeric: tabular-nums; color: #67e8f9; font-weight: 700; }
        .speaker-expert { color: #86efac; font-weight: 700; }
        .speaker-interviewer { color: #cbd5e1; font-weight: 700; }
        .line-target { background: rgba(250, 204, 21, .22); outline: 1px solid rgba(250, 204, 21, .45); }
        .line-row { padding: 4px 8px; border-radius: 6px; }
        .line-no { display: inline-block; width: 52px; color: #64748b; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        .nav-caption { color: #64748b; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border); }

        /* ── Force dark mode on all Streamlit native components ── */
        section[data-testid="stSidebar"] {
          background: #0f172a !important;
          border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * {
          color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] p {
          color: #94a3b8 !important;
        }
        /* Radio button labels */
        div[role="radiogroup"] label, div[role="radiogroup"] p,
        div[data-testid="stRadio"] label, div[data-testid="stRadio"] p {
          color: #f1f5f9 !important;
          font-weight: 600 !important;
        }
        /* Main content area text */
        .stApp, .stApp p, .stApp span, .stApp div,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
          color: var(--text);
        }
        /* Input box */
        .stChatInput textarea, .stTextInput input, .stTextArea textarea {
          background: #182235 !important;
          color: #f8fafc !important;
          border-color: #25324a !important;
        }
        /* Selectbox */
        div[data-testid="stSelectbox"] div {
          background: #182235 !important;
          color: #f8fafc !important;
        }
        /* Streamlit's own light bg overrides */
        .stApp { background-color: var(--bg) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _transcript_label(transcript: TranscriptDocument) -> str:
    return f"{transcript.market} · {transcript.expert_name}"


def _analysis_map(matrix: GroundTruthMatrix) -> dict[str, TranscriptAnalysis]:
    return {analysis.transcript_id: analysis for analysis in matrix.analyses}


def _find_analysis(matrix: GroundTruthMatrix, transcript_id: str) -> TranscriptAnalysis | None:
    return next((analysis for analysis in matrix.analyses if analysis.transcript_id == transcript_id), None)


def _find_question(analysis: TranscriptAnalysis, question_id: str):
    return next((answer for answer in analysis.answers if answer.question_id == question_id), None)


def _load_transcripts() -> list[TranscriptDocument]:
    transcripts: list[TranscriptDocument] = []
    for filename, transcript_id in DEFAULT_TRANSCRIPT_SPECS:
        path = RAW_DIR / filename
        if path.exists():
            transcripts.append(parse_transcript_file(path, transcript_id))
    return transcripts


def _get_extraction_provider():
    if SETTINGS.llm_provider == "openai" or (SETTINGS.openai_api_key and not SETTINGS.llm_provider == "gemini"):
        return OpenAIExtractionProvider(
            model_name=SETTINGS.openai_model,
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
        )
    elif SETTINGS.llm_provider == "gemini" and (SETTINGS.gemini_api_key or SETTINGS.google_api_key):
        return GeminiExtractionProvider(
            api_key=SETTINGS.gemini_api_key or SETTINGS.google_api_key,
            model_name=SETTINGS.gemini_model,
        )
    return MockExtractionProvider()


def _get_synthesis_provider():
    if SETTINGS.llm_provider == "openai" or (SETTINGS.openai_api_key and not SETTINGS.llm_provider == "gemini"):
        return OpenAISynthesisProvider(
            model_name=SETTINGS.openai_model,
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
        )
    elif SETTINGS.llm_provider == "gemini" and (SETTINGS.gemini_api_key or SETTINGS.google_api_key):
        return GeminiSynthesisProvider(
            api_key=SETTINGS.gemini_api_key or SETTINGS.google_api_key,
            model_name=SETTINGS.gemini_model,
        )
    return MockSynthesisProvider()


def _get_rag_provider():
    if SETTINGS.llm_provider == "openai" or (SETTINGS.openai_api_key and not SETTINGS.llm_provider == "gemini"):
        return OpenAIRAGProvider(
            model_name=SETTINGS.openai_model,
            api_key=SETTINGS.openai_api_key,
            base_url=SETTINGS.openai_base_url,
        )
    elif SETTINGS.llm_provider == "gemini" and (SETTINGS.gemini_api_key or SETTINGS.google_api_key):
        return GeminiRAGProvider(
            api_key=SETTINGS.gemini_api_key or SETTINGS.google_api_key,
            model_name=SETTINGS.gemini_model,
        )
    return MockRAGProvider()


def _load_base_matrix(transcripts: list[TranscriptDocument]) -> GroundTruthMatrix:
    if MATRIX_PATH.exists():
        try:
            matrix = load_ground_truth_matrix(MATRIX_PATH)
            expected = {t.transcript_id: t.file_hash for t in transcripts}
            if matrix.source_hashes == expected:
                return matrix
        except Exception:
            pass

    guide = GUIDE_PATH.read_text(encoding="utf-8")
    provider = _get_extraction_provider()
    try:
        return build_ground_truth_matrix(transcripts, guide, provider)
    except Exception:
        return build_ground_truth_matrix(transcripts, guide, MockExtractionProvider())


def _load_synthesis(matrix: GroundTruthMatrix) -> SynthesisReport:
    if SYNTHESIS_PATH.exists():
        try:
            report = load_synthesis_report(SYNTHESIS_PATH)
            if synthesis_cache_is_valid(report, matrix):
                return report
            return report
        except Exception:
            pass

    provider = _get_synthesis_provider()
    try:
        return build_synthesis_report(matrix, provider)
    except Exception:
        return build_synthesis_report(matrix, MockSynthesisProvider())


def _build_rag_engine(transcripts: list[TranscriptDocument]) -> RAGEngine:
    chunks = chunk_transcripts(transcripts)
    embedding_provider = HashEmbeddingProvider()
    if SETTINGS.embedding_provider == "sentence_transformers":
        try:
            embedding_provider = SentenceTransformerEmbeddingProvider(SETTINGS.embedding_model)
            embedding_provider.embed(["startup probe"])
        except Exception:
            embedding_provider = HashEmbeddingProvider()

    retriever = HybridRetriever(
        chunks,
        vector_index=InMemoryVectorIndex(chunks, embedding_provider),
        embedding_provider=embedding_provider,
        top_k_each=SETTINGS.retrieval_candidate_k,
        rrf_k=SETTINGS.rrf_k,
    )
    provider = _get_rag_provider()
    return RAGEngine(
        retriever=retriever,
        transcripts=transcripts,
        provider=provider,
        evidence_threshold=SETTINGS.evidence_gate_threshold,
        top_k=SETTINGS.retrieval_top_k,
        max_evidence=4,
    )


def _initialise_state() -> None:
    if "transcripts" not in st.session_state:
        st.session_state.transcripts = _load_transcripts()
    if "matrix" not in st.session_state:
        st.session_state.matrix = _load_base_matrix(st.session_state.transcripts)
    if "synthesis" not in st.session_state:
        st.session_state.synthesis = _load_synthesis(st.session_state.matrix)
    if "screen" not in st.session_state:
        st.session_state.screen = "MATRIX"
    if "evidence_drawer" not in st.session_state:
        st.session_state.evidence_drawer = None
    if "inspector_transcript_id" not in st.session_state:
        st.session_state.inspector_transcript_id = (
            st.session_state.transcripts[0].transcript_id if st.session_state.transcripts else None
        )
    if "inspector_line" not in st.session_state:
        st.session_state.inspector_line = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = _build_rag_engine(st.session_state.transcripts)


def _navigate(screen: str) -> None:
    st.session_state.screen = screen
    st.session_state.evidence_drawer = None


def _open_evidence(citation: QuoteCitation) -> None:
    st.session_state.evidence_drawer = citation.model_dump(mode="json")


def _jump_to_inspector(citation: QuoteCitation) -> None:
    st.session_state.inspector_transcript_id = citation.transcript_id
    st.session_state.inspector_line = citation.start_line
    st.session_state.screen = "INSPECTOR"
    st.session_state.evidence_drawer = citation.model_dump(mode="json")


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">TRANSCRIPT INTELLIGENCE PRO</div>
          <div class="hero-sub">Closed-Loop Document Intelligence for Qualitative Commercial Due Diligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_navigation() -> None:
    labels = ["MATRIX", "SYNTHESIS", "AI CHAT", "INSPECTOR", "UPLOAD"]
    selected = st.radio(
        "Workspace",
        labels,
        index=labels.index(st.session_state.screen) if st.session_state.screen in labels else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.screen = selected


def _status_badge(status: str) -> str:
    color, label = STATUS_STYLES.get(status, ("#64748b", status))
    return (
        f'<span style="display:inline-block;padding:.18rem .48rem;border-radius:999px;'
        f'background:{color}20;color:{color};border:1px solid {color}55;font-size:.72rem;font-weight:700">'
        f"{html.escape(label)}</span>"
    )


def _render_metrics() -> None:
    transcripts = st.session_state.transcripts
    matrix = st.session_state.matrix
    total_cells = sum(len(a.answers) for a in matrix.analyses)
    verified = sum(
        1
        for analysis in matrix.analyses
        for answer in analysis.answers
        for citation in answer.citations
        if citation.evidence_status.value in {"EXACT_VERIFIED", "NORMALIZED_VERIFIED"}
    )
    cols = st.columns(4)
    cards = [
        (str(len(transcripts)), "Source transcripts"),
        (f"{len(matrix.questions)} × {len(matrix.analyses)}", "Ground-truth cells"),
        (str(verified), "Verified citations"),
        ("RRF 60", "Hybrid retrieval"),
    ]
    for col, (value, label) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )


def _render_evidence_drawer() -> None:
    evidence = st.session_state.evidence_drawer
    if not evidence:
        return
    st.sidebar.markdown("### Evidence Drawer")
    st.sidebar.markdown(
        f'<div class="pill">{html.escape(evidence.get("timestamp", ""))}</div>'
        f'<div class="pill">{html.escape(evidence.get("expert_name", ""))}</div>'
        f'<div class="pill">{html.escape(evidence.get("transcript_id", ""))}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(_status_badge(evidence.get("evidence_status", "")), unsafe_allow_html=True)
    st.sidebar.markdown("**Verbatim source quote**")
    st.sidebar.markdown(f'> {evidence.get("quote", "")}', unsafe_allow_html=False)
    st.sidebar.caption(
        f'Lines {evidence.get("start_line", "?")}–{evidence.get("end_line", "?")} · '
        f'chars {evidence.get("start_char", "?")}–{evidence.get("end_char", "?")}'
    )
    if st.sidebar.button("Jump to Line in Inspector", key="drawer_jump"):
        citation = QuoteCitation.model_validate(evidence)
        _jump_to_inspector(citation)
        st.rerun()
    if st.sidebar.button("Close Evidence Drawer", key="drawer_close"):
        st.session_state.evidence_drawer = None
        st.rerun()


def _render_matrix() -> None:
    matrix = st.session_state.matrix
    st.subheader("Comparative Ground-Truth Matrix")
    st.caption("Cached source answers are rendered directly from the verified matrix; opening a citation does not incur an LLM call.")
    analyses = matrix.analyses
    if not analyses:
        st.warning("No transcript analyses are loaded.")
        return

    header = st.columns(len(analyses) + 1)
    header[0].markdown("**Question**")
    for index, analysis in enumerate(analyses, start=1):
        header[index].markdown(f"**{analysis.market}**  \n{analysis.expert_name}")

    for question in matrix.questions:
        cells = st.columns(len(analyses) + 1)
        cells[0].markdown(f"**{question.question_id}**  \n{question.question_text}")
        for col_index, analysis in enumerate(analyses, start=1):
            answer = _find_question(analysis, question.question_id)
            with cells[col_index]:
                if answer is None:
                    st.info("No answer")
                    continue
                st.markdown(f'<div class="answer-box">{html.escape(answer.answer)}</div>', unsafe_allow_html=True)
                if answer.citations:
                    for c_index, citation in enumerate(answer.citations, start=1):
                        label = f"[{citation.timestamp}]"
                        if st.button(label, key=f"matrix_{analysis.transcript_id}_{question.question_id}_{c_index}"):
                            _open_evidence(citation)
                            st.rerun()
                        st.markdown(_status_badge(citation.evidence_status.value), unsafe_allow_html=True)
                else:
                    st.caption("No verified citation")
                st.caption(f"Confidence: {answer.confidence}")


def _growth_card(title: str, entries: list[str], accent: str) -> None:
    color = accent
    st.markdown(
        f'<div class="panel-card"><div style="color:{color};font-size:.75rem;font-weight:800;letter-spacing:.08em">{html.escape(title.upper())}</div>',
        unsafe_allow_html=True,
    )
    if entries:
        for entry in entries:
            st.markdown(f"- {html.escape(entry)}")
    else:
        st.caption("No classified markets in the cached report.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_synthesis() -> None:
    report = st.session_state.synthesis
    st.subheader("Cross-Expert Synthesis")
    st.caption("Themes and growth classifications are displayed from the current synthesis report; no new LLM request is made while navigating.")

    left, right = st.columns(2)
    with left:
        st.markdown("### Shared Consensus Themes")
        for index, theme in enumerate(report.consensus_themes, start=1):
            with st.expander(f"{index}. {theme.title}", expanded=index == 1):
                st.write(theme.description)
                for market, evidence in theme.evidence_by_country.items():
                    st.markdown(f"**{market}**")
                    for item in evidence:
                        st.markdown(f"> {item}")
    with right:
        st.markdown("### Strategic Disagreements")
        for index, disagreement in enumerate(report.disagreements, start=1):
            with st.expander(f"{index}. {disagreement.topic}", expanded=index == 1):
                st.write(disagreement.description)
                for stakeholder, stance in disagreement.stances_by_stakeholder.items():
                    st.markdown(f"**{stakeholder}** — {stance}")

    st.markdown("### 3–5 Year Market Growth Spectrum")
    a, b, c = st.columns(3)
    with a:
        _growth_card("Conservative", report.growth_spectrum.conservative, "#60a5fa")
    with b:
        _growth_card("Moderate", report.growth_spectrum.moderate, "#22d3ee")
    with c:
        _growth_card("Bullish", report.growth_spectrum.bullish, "#10b981")


def _progressive_answer(text: str) -> None:
    placeholder = st.empty()
    words = text.split()
    if not words:
        placeholder.write("")
        return
    rendered: list[str] = []
    for word in words:
        rendered.append(word)
        placeholder.markdown(" ".join(rendered))
        time.sleep(0.008)


def _render_chat_citation(citation: QuoteCitation, index: int) -> None:
    with st.expander(f"Evidence {index} · [{citation.timestamp}] {citation.expert_name}"):
        st.markdown(_status_badge(citation.evidence_status.value), unsafe_allow_html=True)
        st.markdown(f"> {citation.quote}")
        st.caption(f"Transcript {citation.transcript_id} · lines {citation.start_line}–{citation.end_line}")
        if st.button("Open Evidence Drawer", key=f"chat_ev_{citation.turn_id}_{index}"):
            _open_evidence(citation)
            st.rerun()
        if st.button("Jump to Inspector", key=f"chat_jump_{citation.turn_id}_{index}"):
            _jump_to_inspector(citation)
            st.rerun()


def _render_chat() -> None:
    st.subheader("Cross-Transcript Conversational RAG")
    st.caption("Hybrid lexical + vector retrieval with RRF fusion, an evidence gate, deterministic source resolution, and abstention on unsupported questions.")

    with st.sidebar:
        st.markdown("### Retrieval Filters")
        markets = ["All"] + sorted({t.market for t in st.session_state.transcripts})
        market = st.selectbox("Market", markets, key="chat_market")
        if market != "All":
            available_roles = {t.role for t in st.session_state.transcripts if t.market == market}
        else:
            available_roles = {t.role for t in st.session_state.transcripts}
        role_values = ["All"] + sorted(available_roles)
        role = st.selectbox("Role", role_values, key="chat_role")
        st.caption(f"Evidence gate threshold: {SETTINGS.evidence_gate_threshold:.2f}")

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item.get("abstained"):
                st.warning("Abstained: the available transcripts did not provide sufficient grounded evidence.")
            for idx, citation in enumerate(item.get("citations", []), start=1):
                _render_chat_citation(QuoteCitation.model_validate(citation), idx)

    query = st.chat_input("Ask a question across the expert transcripts…")
    if not query:
        return

    selected_market = None if market == "All" else market
    selected_role = None if role == "All" else role
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        try:
            try:
                response: RAGResponse = st.session_state.rag_engine.answer(
                    query,
                    market=selected_market,
                    role=selected_role,
                )
            except Exception as live_exc:
                st.caption(f"ℹ️ Live LLM ({live_exc}); answered using deterministic grounded fallback.")
                fallback_engine = RAGEngine(
                    retriever=st.session_state.rag_engine.retriever,
                    transcripts=st.session_state.transcripts,
                    provider=MockRAGProvider(),
                    evidence_threshold=SETTINGS.evidence_gate_threshold,
                    top_k=SETTINGS.retrieval_top_k,
                    max_evidence=4,
                )
                response = fallback_engine.answer(
                    query,
                    market=selected_market,
                    role=selected_role,
                )

            _progressive_answer(response.answer)
            citations = st.session_state.rag_engine.citations_for_response(response)
            if response.abstained:
                st.warning("Abstained: the evidence gate did not find sufficient source support.")
            for idx, citation in enumerate(citations, start=1):
                _render_chat_citation(citation, idx)
            st.session_state.chat_history.append(
                {
                    "query": query,
                    "answer": response.answer,
                    "abstained": response.abstained,
                    "citations": [citation.model_dump(mode="json") for citation in citations],
                }
            )
        except Exception as exc:
            st.error(f"RAG request failed: {exc}")


def _render_transcript_html(transcript: TranscriptDocument, target_line: int | None) -> str:
    rows: list[str] = []
    target_lines = set()
    if target_line is not None:
        target_lines = {target_line}
        for turn in transcript.turns:
            if turn.start_line <= target_line <= turn.end_line:
                target_lines.update(range(turn.start_line, min(turn.end_line, target_line + 2) + 1))
                break

    raw_lines = transcript.raw_text.splitlines()
    turn_by_line: dict[int, tuple[str, str]] = {}
    for turn in transcript.turns:
        for line_no in range(turn.start_line, turn.end_line + 1):
            turn_by_line[line_no] = (turn.timestamp, turn.speaker)

    for line_no, raw_line in enumerate(raw_lines, start=1):
        timestamp, speaker = turn_by_line.get(line_no, ("", ""))
        klass = " line-target" if line_no in target_lines else ""
        safe = html.escape(raw_line)
        if timestamp:
            speaker_klass = "speaker-interviewer" if "interviewer" in speaker.casefold() else "speaker-expert"
            prefix = f'<span class="timestamp">[{html.escape(timestamp)}]</span> <span class="{speaker_klass}">{html.escape(speaker)}</span> '
            if ":" in safe:
                _, remainder = safe.split(":", 1)
                safe = prefix + html.escape(remainder.lstrip())
            else:
                safe = prefix + safe
        rows.append(f'<div id="line-{line_no}" class="line-row{klass}"><span class="line-no">{line_no:04d}</span>{safe}</div>')

    target_script = ""
    if target_line is not None:
        target_script = f"<script>setTimeout(function(){{var e=document.getElementById('line-{target_line}');if(e)e.scrollIntoView({{behavior:'smooth',block:'center'}})}},150);</script>"
    return "<div>" + "".join(rows) + "</div>" + target_script


def _render_inspector() -> None:
    st.subheader("Raw Transcript Inspector")
    transcripts = st.session_state.transcripts
    if not transcripts:
        st.warning("No transcripts available.")
        return

    labels = {_transcript_label(t): t.transcript_id for t in transcripts}
    current_id = st.session_state.inspector_transcript_id or transcripts[0].transcript_id
    current = next((t for t in transcripts if t.transcript_id == current_id), transcripts[0])
    selected_label = st.selectbox(
        "Active transcript",
        list(labels.keys()),
        index=list(labels.values()).index(current.transcript_id),
    )
    current = next(t for t in transcripts if t.transcript_id == labels[selected_label])
    st.session_state.inspector_transcript_id = current.transcript_id

    if st.session_state.inspector_line is not None:
        st.info(
            f"Jump target: line {st.session_state.inspector_line}. The target turn is highlighted below."
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Turns", current.total_turns)
    with col2:
        st.metric("Duration", current.duration_str)
    with col3:
        st.metric("SHA-256", current.file_hash[:12] + "…")

    st.html(_render_transcript_html(current, st.session_state.inspector_line), unsafe_allow_javascript=True)
    if st.session_state.inspector_line is not None:
        if st.button("Clear Jump Highlight"):
            st.session_state.inspector_line = None
            st.rerun()


def _unique_uploaded_id(filename: str, raw_bytes: bytes) -> str:
    digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
    stem = Path(filename).stem.lower().replace(" ", "_")
    return f"upload_{stem}_{digest}"


def _process_upload(uploaded_file) -> None:
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")
    if not uploaded_file.name.lower().endswith(".txt"):
        raise ValueError("Only .txt transcript files are supported.")

    file_hash = compute_sha256(raw_bytes)
    raw_text = raw_bytes.decode("utf-8")
    transcript_id = _unique_uploaded_id(uploaded_file.name, raw_bytes)
    existing = {t.file_hash for t in st.session_state.transcripts}
    if file_hash in existing:
        raise ValueError("This exact file is already loaded; SHA-256 idempotency prevented a duplicate.")

    transcript = parse_transcript_text(
        raw_text,
        transcript_id=transcript_id,
        filename=uploaded_file.name,
        file_hash=file_hash,
    )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / uploaded_file.name
    target.write_bytes(raw_bytes)

    transcripts = [*st.session_state.transcripts, transcript]
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    provider = _get_extraction_provider()
    with st.spinner("Parsing, chunking, indexing, and extracting the six-question matrix…"):
        try:
            new_matrix = build_ground_truth_matrix(transcripts, guide, provider)
        except Exception:
            new_matrix = build_ground_truth_matrix(transcripts, guide, MockExtractionProvider())
        try:
            new_synthesis_provider = _get_synthesis_provider()
            new_synthesis = build_synthesis_report(new_matrix, new_synthesis_provider)
        except Exception:
            new_synthesis = build_synthesis_report(new_matrix, MockSynthesisProvider())
        new_rag = _build_rag_engine(transcripts)

    st.session_state.transcripts = transcripts
    st.session_state.matrix = new_matrix
    st.session_state.synthesis = new_synthesis
    st.session_state.rag_engine = new_rag
    st.session_state.inspector_transcript_id = transcript.transcript_id
    st.session_state.inspector_line = None
    st.session_state.screen = "MATRIX"
    st.success(f"Loaded {uploaded_file.name} as {transcript.market} · {transcript.expert_name}.")


def _render_upload() -> None:
    st.subheader("Document Upload & Dynamic Ingestion")
    st.write("Drop a timestamped `.txt` expert transcript below. The parser validates required metadata and timestamps before it enters the session workspace.")
    uploaded = st.file_uploader("Transcript file", type=["txt"], accept_multiple_files=False)

    if uploaded is not None:
        st.markdown(
            f'<div class="panel-card"><strong>{html.escape(uploaded.name)}</strong><br>'
            f'SHA-256: <code>{hashlib.sha256(uploaded.getvalue()).hexdigest()}</code></div>',
            unsafe_allow_html=True,
        )
        if st.button("Ingest transcript", type="primary"):
            try:
                _process_upload(uploaded)
                st.rerun()
            except UnicodeDecodeError:
                st.error("The file is not valid UTF-8 text.")
            except Exception as exc:
                st.error(f"Upload rejected: {exc}")

    st.markdown("### Active workspace sources")
    for transcript in st.session_state.transcripts:
        st.markdown(
            f'- **{transcript.market}** — {transcript.expert_name} · {transcript.role} · `{transcript.file_hash[:12]}…`'
        )

    provider = _get_extraction_provider()
    provider_label = f"{provider.__class__.__name__} ({getattr(provider, 'model_name', 'mock')})"
    st.info(
        f"Active extraction & synthesis provider: **{provider_label}**. "
        "Credentials loaded from `.env` (Gemini / OpenAI / Mock fallback)."
    )


def main() -> None:
    inject_css()
    _initialise_state()
    _render_header()
    _render_navigation()
    _render_evidence_drawer()

    if st.session_state.screen == "MATRIX":
        _render_metrics()
        _render_matrix()
    elif st.session_state.screen == "SYNTHESIS":
        _render_metrics()
        _render_synthesis()
    elif st.session_state.screen == "AI CHAT":
        _render_chat()
    elif st.session_state.screen == "INSPECTOR":
        _render_inspector()
    elif st.session_state.screen == "UPLOAD":
        _render_upload()


if __name__ == "__main__":
    main()
