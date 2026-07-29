import os
import streamlit as st
import weave

from inference.predictor import load_tokenizer_and_model, get_prediction
from inference.logger import log_inference

weave.init(os.getenv("WANDB_PROJECT", "flan-t5-dialogsum"))

@st.cache_resource
def initialize_model():
    try:
        load_tokenizer_and_model()
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.stop()

initialize_model()

st.set_page_config(
    page_title="Text Summarization",
    page_icon="📑",
)

st.title("📑 Text Summarization")

st.write("Generate summaries using a fine tuned FLAN-T5 model.")

text = st.text_area(
    "Enter Text",
    height=300,
    placeholder="Paste your article, dialogue or document here..."
)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text")

    else:
        summary = get_prediction(text, max_new_tokens=128)

        log_inference(
            query=text,
            response=summary,
            model=os.getenv("HF_MODEL_ID", "google/flan-t5-base")
        )

        st.success("Summary Generated Successfully!")
        
        st.text_area(
            "📑 Generated Summary",
            value=summary,
            height=180,
            disabled=True,
        )