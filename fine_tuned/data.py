from datasets import load_dataset
from transformers import AutoTokenizer

model = "google/flan-t5-base"
input_max_length = 512
output_max_length = 128

def loading_dataset():
    return load_dataset("knkarthick/dialogsum")


def loading_tokenizer():
    return AutoTokenizer.from_pretrained(model)


def tokenization(example, tokenizer):
    inputs = ["summarize: " + dialogue for dialogue in example["dialogue"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length = input_max_length,
        padding = "max_length",
        truncation = True
    )

    labels = tokenizer(
        text_target = example["summary"],
        max_length = output_max_length,
        padding = "max_length",
        truncation = True
    )

    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in tokens]
          for tokens in labels["input_ids"]
    ]
    return model_inputs


def tokenize_data(tokenizer = None):
    if tokenizer is None:
        tokenizer = loading_tokenizer()
        
    dataset = loading_dataset()
    tokenized_dataset = dataset.map(
        lambda example : tokenization(example, tokenizer),
        batched = True,
        remove_columns = dataset["train"].column_names
    )
    return tokenized_dataset