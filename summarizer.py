# summarizer.py
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn"
)


def summarize_text(text: str, max_len: int = 130, min_len: int = 30) -> str:
    if not text or len(text.strip()) == 0:
        return "No text provided."

    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary[0]["summary_text"]

