import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def upload_model(model_path, repo_name):
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN not found")
    
    login(token = token)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model uploaded: https://huggingface.co/{repo_name}")

if __name__ =="__main__":
    if len(sys.argv) != 3:
        print("Missing required arguments")
        print("Provide:")
        print("Saved model directory")
        print("Hugging Face repository name")
        sys.exit(1)

    upload_model(sys.argv[1], sys.argv[2])