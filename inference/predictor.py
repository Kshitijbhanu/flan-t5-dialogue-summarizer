import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def device_selection():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = device_selection()

model = None
tokenizer = None


def load_tokenizer_and_model():
    global model, tokenizer
    if model is not None:
        return tokenizer
    
    model_path = os.getenv("HF_MODEL_ID", "google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer


def get_prediction(text, max_new_tokens = 128):
    load_tokenizer_and_model()

    input = ["summarize :" + text]

    inputs = tokenizer(
        input,
        return_tensors = "pt",
        truncation = True,
        max_length = 512
    )
    inputs = {k : v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens = max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens = True)
