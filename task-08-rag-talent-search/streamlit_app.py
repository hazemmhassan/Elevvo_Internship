"""Recruiter chat interface for the privacy-aware talent search pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from talent_search.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    create_local_embeddings,
)
from talent_search.llm_evaluation import create_candidate_evaluation_chain  # noqa: E402
from talent_search.providers import (  # noqa: E402
    ProviderConfigurationError,
    create_chat_model,
    resolve_llm_settings,
)
from talent_search.service import TalentSearchResult, search_talent  # noqa: E402
from talent_search.vector_store import load_faiss_index  # noqa: E402


INDEX_DIRECTORY = PROJECT_ROOT / ".artifacts" / "faiss_index"


@st.cache_resource(show_spinner="Loading the local embedding model and index...")
def load_search_index():
    embeddings = create_local_embeddings(DEFAULT_EMBEDDING_MODEL)
    return load_faiss_index(
        INDEX_DIRECTORY,
        embeddings,
        expected_embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    )


@st.cache_resource(show_spinner=False)
def load_optional_evaluator():
    try:
        settings = resolve_llm_settings()
    except ProviderConfigurationError:
        return None, None
    model = create_chat_model(settings)
    return create_candidate_evaluation_chain(model), settings.model


def render_search_result(result: TalentSearchResult) -> None:
    if result.query_safety.was_modified:
        removed = ", ".join(result.query_safety.protected_terms)
        st.warning(
            f"Protected demographic terms were excluded before search: {removed}. "
            f"Job-focused query used: “{result.query_safety.cleaned_query}”"
        )

    if not result.matches:
        st.info("No candidates were returned for this query.")
        return

    assessments = {
        assessment.candidate_id: assessment
        for assessment in (result.evaluation.assessments if result.evaluation else ())
    }
    for rank, match in enumerate(result.matches, start=1):
        st.subheader(f"#{rank} · {match.candidate_id}")
        coverage_columns = st.columns(3)
        coverage_columns[0].metric(
            "Required-term coverage",
            f"{(match.term_coverage or 0):.0%}",
        )
        coverage_columns[1].metric(
            "Phrase coverage",
            f"{(match.phrase_coverage or 0):.0%}",
        )
        coverage_columns[2].metric(
            "Dense similarity",
            f"{(match.dense_score or 0):.3f}",
        )
        st.caption(
            "These are retrieval diagnostics, not a probability of job fit or success."
        )

        assessment = assessments.get(match.candidate_id)
        if assessment:
            st.write(assessment.summary)
            if assessment.strengths:
                st.markdown("**Supported strengths**")
                for strength in assessment.strengths:
                    st.markdown(f"- {strength}")
            if assessment.gaps:
                st.markdown("**Missing or unverified**")
                for gap in assessment.gaps:
                    st.markdown(f"- {gap}")
            st.caption(f"Uncertainty: {assessment.uncertainty}")

        with st.expander("View anonymized supporting evidence"):
            for evidence in match.evidence:
                section = str(evidence.document.metadata["section"])
                st.markdown(f"**{section}**")
                st.text(evidence.document.page_content)

    if result.bias_audit.passed:
        st.success("Privacy and sensitive-metadata checks passed for these results.")
    else:
        for flag in result.bias_audit.flags:
            st.error(flag)
    st.caption(result.bias_audit.disclaimer)


st.set_page_config(
    page_title="RAG Talent Search",
    page_icon="🔎",
    layout="wide",
)
st.title("RAG-Powered Talent Search")
st.write(
    "Ask for job-related skills or experience. The system retrieves three anonymous "
    "candidates and shows the resume evidence behind each result."
)

with st.sidebar:
    st.header("Pipeline status")
    st.write(f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`")
    st.write("Vector database: FAISS (local)")
    st.write("Ranking: dense + BM25 + constraint coverage")
    if os.getenv("OPENAI_API_KEY", "").strip():
        st.success("LLM explanations configured")
    else:
        st.info("Retrieval mode: add `OPENAI_API_KEY` for LLM explanations")
    st.divider()
    st.caption(
        "Decision-support demo only. A recruiter must verify evidence and use a "
        "job-related, lawful hiring process."
    )

if not INDEX_DIRECTORY.is_dir():
    st.error(
        "The local index is missing. Run `python scripts/build_index.py <dataset>` "
        "before starting the app."
    )
    st.stop()

try:
    vector_store = load_search_index()
except Exception as exc:
    st.error(f"The local index could not be loaded safely: {exc}")
    st.stop()

evaluator_chain, evaluator_model = load_optional_evaluator()
if evaluator_model:
    st.caption(f"LLM explanation model: {evaluator_model}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_search_result(message["content"])

prompt = st.chat_input(
    "Example: Find a data analyst with SQL, Tableau, and dashboard experience"
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching anonymized resume evidence..."):
                result = search_talent(
                    vector_store,
                    prompt,
                    evaluator_chain=evaluator_chain,
                )
            render_search_result(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result}
            )
        except Exception as exc:
            st.error(str(exc))
