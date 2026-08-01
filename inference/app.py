import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from pydantic import ConfigDict, BaseModel
from contextlib import asynccontextmanager
import weave

from inference.predictor import load_tokenizer_and_model, get_prediction
from inference.logger import log_inference

MODEL_NAME = os.environ.get("HF_MODEL_ID", "google/flan-t5-base")

class SummarizeRequest(BaseModel):
    model_config = ConfigDict(strict = False)
    text : str
    max_length : int = 128

class SummarizeResponse(BaseModel):
    summary : str

@asynccontextmanager
async def lifespan(app : FastAPI):
    wandb_project =  os.environ.get("WANDB_PROJECT", "flan-t5-dialogsum")
    weave.init(project_name = wandb_project)
    app.state.model = load_tokenizer_and_model()
    yield

app = FastAPI(title="flan-t5-dialogsum", version= "1.0.0", lifespan= lifespan)

@app.get("/", include_in_schema = False)
async def home():
    return RedirectResponse(url = "/docs")

@app.get("/health")
async def health():
    return {"status": "OK"}

@app.post("/summarize", response_model = SummarizeResponse)
async def summarize(request : SummarizeRequest):
    summary = get_prediction(request.text, max_new_tokens=request.max_length)
    log_inference(query = request.text, response = summary, model = MODEL_NAME)
    return SummarizeResponse(summary = summary)

print("flan-t5-dialogsum Fasapi")