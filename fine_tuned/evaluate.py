import torch
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from datasets import load_dataset

pre_model = "google/flan-t5-base"

input_token_length = 512
output_tokens = 128


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def generate_summary(text, tokenizer, model, max_new_tokens = output_tokens):
    inputs = ["summarize: "+ text]
    
    model_input = tokenizer(
        inputs,
        max_length = input_token_length,
        return_tensors = "pt",
        truncation = True
    )

    with torch.no_grad():
        output = model.generate(**model_input, max_new_tokens = max_new_tokens)
        return tokenizer.decode(output[0], skip_special_tokens = True)


def rouge_score(dialogues, tokenizer, model, max_new_tokens = output_tokens):
    rouge = evaluate.load("rouge")
    predictions = [generate_summary(dial["dialogue"], tokenizer, model, max_new_tokens) for dial in dialogues]
    references = [dial["summary"] for dial in dialogues]
    score = rouge.compute(predictions = predictions, references = references)
    return {k : v for k, v in score.items()}


def eval_basemodel(samples = 200):
    test_sample = load_dataset("knkarthick/dialogsum", split = f"test[:{samples}]")
    tokenizer, model = load_model_and_tokenizer(pre_model)
    score = rouge_score(test_sample, tokenizer, model)
    for k, v in score.items():
        print(f"{k} : {v}")
    return score


def eval_finetuned(model_path, samples = 200):
    test_sample = load_dataset("knkarthick/dialogsum", split = f"test[:{samples}]")
    tokenizer, model = load_model_and_tokenizer(model_path)
    score = rouge_score(test_sample, tokenizer, model)
    for k, v in score.items():
        print(f"{k} : {v}")
    return score


if __name__ == "__main__":
    model_path = "saved_models/flan-t5-dialogsum"
    rouge_baseline = eval_basemodel()
    rouge_finetuned = eval_finetuned(model_path)
    for k in rouge_baseline:
        improvement = rouge_finetuned[k] - rouge_baseline[k]
        print(f"{k} : {improvement:+.2f}")