"""Streamlit interface for local extractive question answering."""

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from question_answering.model import DEFAULT_MODEL, create_question_answerer
from question_answering.ui import highlight_context


MODELS = (
    DEFAULT_MODEL,
    "distilbert/distilbert-base-cased-distilled-squad",
)
EXAMPLE_CONTEXT = (
    "The Nile is a major north-flowing river in northeastern Africa. "
    "It flows into the Mediterranean Sea. The Nile is approximately "
    "6,650 kilometres long."
)
EXAMPLE_QUESTION = "Where does the Nile flow?"


@st.cache_resource(show_spinner=False)
def load_answerer(model_name: str, device: str):
    """Load and cache model weights only after the user requests an answer."""
    return create_question_answerer(model_name, device=device)


st.set_page_config(
    page_title="Extractive Question Answering",
    page_icon="❓",
    layout="wide",
)
st.title("Extractive Question Answering")
st.markdown(
    "Ask a question about a passage. A local transformer finds the answer "
    "directly inside the supplied context—no paid API or external LLM call."
)

with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox("Question-answering model", MODELS)
    selected_device = st.selectbox(
        "Compute device",
        ("auto", "cpu", "cuda"),
        help="Auto uses CUDA when the installed PyTorch build supports it.",
    )
    st.divider()
    st.markdown(
        "**Pipeline**\n\n"
        "1. Tokenize question and context\n"
        "2. Score possible start/end tokens\n"
        "3. Extract the highest-scoring context span\n"
        "4. Display confidence and character offsets"
    )

context = st.text_area("Context passage", value=EXAMPLE_CONTEXT, height=220)
question = st.text_area("Question", value=EXAMPLE_QUESTION, height=90)

if st.button("Extract answer", type="primary", use_container_width=True):
    if not context.strip() or not question.strip():
        st.warning("Enter both a context passage and a question.")
    else:
        try:
            with st.spinner("Loading the local model and extracting an answer..."):
                answerer = load_answerer(selected_model, selected_device)
                prediction = answerer.answer(question, context)
        except Exception as error:
            st.error(f"The answer could not be extracted: {error}")
        else:
            st.subheader("Answer")
            st.success(prediction.answer)
            confidence, span, model = st.columns(3)
            confidence.metric("Confidence", f"{prediction.score:.1%}")
            span.metric("Character span", f"{prediction.start}:{prediction.end}")
            model.metric("Model", prediction.model_name.split("/")[-1])
            st.subheader("Answer in context")
            st.markdown(
                highlight_context(context, prediction.start, prediction.end),
                unsafe_allow_html=True,
            )

st.caption(
    "Extractive QA can only return text found in the context. Review low-confidence "
    "answers and use the benchmark report to compare model quality."
)
