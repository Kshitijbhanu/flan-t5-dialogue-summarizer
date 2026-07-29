import torch
from fine_tuned.data import model, loading_tokenizer, tokenize_data

from transformers import (
    AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, DataCollatorForSeq2Seq
)


def train_model():
    tokenizer = loading_tokenizer()
    tokenized_dataset = tokenize_data(tokenizer)
    model_name = AutoModelForSeq2SeqLM.from_pretrained(model)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer = tokenizer,
        model = model_name
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir= "saved_models/flan-t5-dialogsum",

        learning_rate= 2e-5,
        per_device_train_batch_size= 8,
        per_device_eval_batch_size= 8,
        num_train_epochs= 2,
        weight_decay= 0.01,

        eval_strategy= "epoch",
        save_strategy= "epoch",
        load_best_model_at_end= True,
        predict_with_generate= True,

        bf16= use_bf16,
        fp16= use_fp16,

        logging_steps= 200,
        logging_dir= "saved_models/logs",
        report_to= "none"
    )

    trainer = Seq2SeqTrainer(
        processing_class = tokenizer,
        model = model_name,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["validation"],
        data_collator = data_collator
    )

    trainer.train()
    trainer.save_model("saved_models/flan-t5-dialogsum")
    tokenizer.save_pretrained("saved_models/flan-t5-dialogsum")
    return trainer

if __name__ == "__main__":
    train_model()