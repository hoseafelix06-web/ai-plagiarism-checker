import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Plagiarism Checker",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):
    """Normalize text for comparison."""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# SIMILARITY CALCULATION
# ==========================================

def calculate_similarity(document_a, document_b):

    text_a = preprocess_text(document_a)
    text_b = preprocess_text(document_b)

    if not text_a or not text_b:
        return 0.0

    vectorizer = TfidfVectorizer()

    try:
        vectors = vectorizer.fit_transform(
            [text_a, text_b]
        )

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(score * 100)

    except ValueError:
        return 0.0


# ==========================================
# MATCHING SENTENCE DETECTION
# ==========================================

def find_matching_sentences(document_a, document_b):

    sentences_a = re.split(
        r"(?<=[.!?])\s+",
        document_a
    )

    sentences_b = re.split(
        r"(?<=[.!?])\s+",
        document_b
    )

    normalized_b = {
        preprocess_text(sentence)
        for sentence in sentences_b
        if preprocess_text(sentence)
    }

    matches = []

    for sentence in sentences_a:

        normalized = preprocess_text(sentence)

        if normalized and normalized in normalized_b:
            matches.append(sentence.strip())

    return matches


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("📄 AI-Based Plagiarism Checker")

st.markdown(
    """
    **An Intelligent Text Similarity Detection System**

    This application compares two text documents using
    Natural Language Processing (NLP), TF-IDF, and
    cosine similarity.
    """
)

st.info(
    "This educational prototype compares documents "
    "provided by the user. It does not search the "
    "internet or independently establish plagiarism."
)


# ==========================================
# DOCUMENT INPUT
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Document A")

    text_a = st.text_area(
        "Enter the first document:",
        height=250,
        key="text_a"
    )

    file_a = st.file_uploader(
        "Or upload Document A (.txt)",
        type=["txt"],
        key="file_a"
    )


with col2:

    st.subheader("Document B")

    text_b = st.text_area(
        "Enter the second document:",
        height=250,
        key="text_b"
    )

    file_b = st.file_uploader(
        "Or upload Document B (.txt)",
        type=["txt"],
        key="file_b"
    )


# ==========================================
# READ UPLOADED FILES
# ==========================================

if file_a is not None:

    try:
        text_a = file_a.getvalue().decode("utf-8")

    except UnicodeDecodeError:
        st.error(
            "Document A could not be read. "
            "Please upload a UTF-8 encoded text file."
        )


if file_b is not None:

    try:
        text_b = file_b.getvalue().decode("utf-8")

    except UnicodeDecodeError:
        st.error(
            "Document B could not be read. "
            "Please upload a UTF-8 encoded text file."
        )


# ==========================================
# COMPARE DOCUMENTS
# ==========================================

if st.button("Check Similarity", type="primary"):

    if not text_a.strip() or not text_b.strip():

        st.warning(
            "Please enter text or upload a valid "
            "text file for both documents."
        )

    else:

        score = calculate_similarity(
            text_a,
            text_b
        )

        matches = find_matching_sentences(
            text_a,
            text_b
        )

        st.divider()

        st.header("Comparison Results")

        st.metric(
            "Text Similarity",
            f"{score:.2f}%"
        )

        st.progress(
            min(max(int(round(score)), 0), 100)
        )

        st.caption(
            "This score measures textual similarity using "
            "TF-IDF and cosine similarity. It is not a "
            "probability of plagiarism."
        )

        st.subheader("Exact Matching Sentences")

        if matches:

            st.success(
                f"{len(matches)} matching sentence(s) found."
            )

            for number, sentence in enumerate(
                matches,
                start=1
            ):

                st.markdown(
                    f"**{number}.** {sentence}"
                )

        else:

            st.write(
                "No exact matching sentences were found."
            )

        st.warning(
            "Review the matching passages, citations, "
            "and context before drawing conclusions "
            "about plagiarism."
        )
