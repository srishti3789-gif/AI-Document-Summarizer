# AI Document Summarizer

An AI-powered PDF document summarizer that extracts text from PDF files and generates concise summaries using Natural Language Processing and TF-IDF.

## Features

* Upload PDF documents
* Extract text from PDFs
* Generate automatic summaries
* Adjust the summary length
* View original extracted text
* Compare original and summarized word counts
* Calculate text reduction percentage

## Technologies Used

* Python
* Streamlit
* PyPDF
* Scikit-learn
* TF-IDF
* Natural Language Processing

## How It Works

The application follows this process:

**PDF → Text Extraction → Sentence Splitting → TF-IDF → Sentence Scoring → Summary**

The application uses TF-IDF to identify important sentences and selects the highest-scoring sentences to create an extractive summary.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
AI-Document-Summarizer/
├── README.md
├── app.py
└── requirements.txt
```

## Note

This project uses extractive summarization. It selects important sentences from the original document rather than generating completely new sentences.
