import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Summarizer")
st.write(
    "Upload a PDF document and generate a concise summary "
    "using Natural Language Processing."
)


# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Split Text into Sentences
# -----------------------------
def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]

    return sentences


# -----------------------------
# Generate Summary
# -----------------------------
def generate_summary(text, summary_percentage):
    sentences = split_into_sentences(text)

    if not sentences:
        return "No readable sentences were found in the document."

    if len(sentences) <= 2:
        return " ".join(sentences)

    # Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return "Unable to generate a summary from this document."

    # Calculate importance score for each sentence
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Number of sentences to include
    number_of_sentences = max(
        1,
        int(len(sentences) * summary_percentage / 100)
    )

    number_of_sentences = min(
        number_of_sentences,
        len(sentences)
    )

    # Rank sentences by importance
    ranked_sentences = sorted(
        range(len(sentences)),
        key=lambda i: sentence_scores[i],
        reverse=True
    )

    # Select the most important sentences
    selected_indices = ranked_sentences[:number_of_sentences]

    # Put selected sentences back into original order
    selected_indices.sort()

    summary = " ".join(
        sentences[i] for i in selected_indices
    )

    return summary


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Extract text
    document_text = extract_text_from_pdf(uploaded_file)

    if not document_text.strip():

        st.error(
            "No readable text was found in this PDF. "
            "Please upload a text-based PDF."
        )

    else:

        # -----------------------------
        # Document Information
        # -----------------------------
        word_count = len(document_text.split())

        st.info(
            f"Document contains approximately {word_count} words."
        )

        # -----------------------------
        # Summary Length
        # -----------------------------
        summary_percentage = st.slider(
            "Choose summary length:",
            min_value=10,
            max_value=50,
            value=25,
            step=5
        )

        st.write(
            f"The summary will contain approximately "
            f"{summary_percentage}% of the original sentences."
        )

        # -----------------------------
        # Generate Summary
        # -----------------------------
        if st.button("✨ Generate Summary"):

            with st.spinner("Analyzing document..."):

                summary = generate_summary(
                    document_text,
                    summary_percentage
                )

            st.subheader("📝 Summary")

            st.write(summary)

            # -----------------------------
            # Summary Statistics
            # -----------------------------
            original_words = len(document_text.split())
            summary_words = len(summary.split())

            reduction = (
                (1 - summary_words / original_words) * 100
                if original_words > 0
                else 0
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Original Words",
                    original_words
                )

            with col2:
                st.metric(
                    "Summary Words",
                    summary_words
                )

            with col3:
                st.metric(
                    "Text Reduction",
                    f"{reduction:.1f}%"
                )

            # -----------------------------
            # View Original Text
            # -----------------------------
            with st.expander("📖 View Extracted Text"):

                st.write(document_text)
