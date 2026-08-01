# FLAN-T5 Dialogue Summarization

A dialogue summarization project built using Google's **FLAN-T5** model.
This project covers the complete NLP workflow from fine tuning a pretrained sequence to sequence model on the DialogSum dataset to deploying it as a production ready inference API using **FastAPI**, **Docker** and **GitHub Actions**.
The repository includes model training, evaluation, Hugging Face model publishing a REST API for inference a Streamlit web application and a CI/CD pipeline for automated Docker image builds.

---

## Project Overview

The objective of this project is to generate concise summaries from multi turn conversations while preserving the important information.

The complete workflow includes:

- Fine tuning FLAN-T5 on the DialogSum dataset
- Evaluating model performance using ROUGE metrics
- Uploading the trained model to Hugging Face
- Serving predictions through a FastAPI REST API
- Building a Streamlit application for interactive testing
- Containerizing the inference service with Docker
- Automating Docker builds and image publishing using GitHub Actions

---

## Project Structure

```text
.
├── fine_tuned/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── huggingface_upload.py
│
├── inference/
│   ├── app.py
│   ├── predictor.py
│   ├── logger.py
│   ├── streamlit.py
│   └── Dockerfile
│
├── .github/
│   └── workflows/
│       └── github_deploy.yml
│
├── saved_models/
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- FastAPI
- Streamlit
- Docker
- GitHub Actions
- AWS Elastic Container Registry (ECR)
- Weights & Biases Weave

---

## Dataset

This project uses the **DialogSum** dataset from Hugging Face.

Each sample consists of:

- A dialogue
- A human written summary

The model learns to generate concise summaries from conversations between multiple speakers.

---

## Model Training

The model is fine tuned using:

- google/flan-t5-base
- Hugging Face Seq2SeqTrainer
- DataCollatorForSeq2Seq
- Teacher forcing
- Cross entropy loss

Training pipeline:

- Load DialogSum dataset
- Tokenize dialogues and summaries
- Fine tune FLAN-T5
- Validate after every epoch
- Save the best performing model

---

## Model Evaluation

Model performance is evaluated using **ROUGE** metrics.

The evaluation compares:

- Pretrained FLAN-T5
- Fine tuned FLAN-T5

This provides an objective comparison of summarization quality after fine tuning.

---

## Inference API

The trained model is served using **FastAPI**.

### Available Endpoints

#### Health Check

```http
GET /health
```

Response

```json
{
  "status": "OK"
}
```

#### Generate Summary

```http
POST /summarize
```

Example Request

```json
{
  "text": "Your dialogue here",
  "max_length": 128
}
```

Example Response

```json
{
  "summary": "Generated summary"
}
```

---

## Example

### Input Dialogue

```text
#Person1#: Hi Sarah, are we still meeting tomorrow to discuss the project?

#Person2#: Yes. I booked the conference room from 10 AM to 11 AM.

#Person1#: Great. I'll prepare the presentation slides with the latest sales figures and customer feedback.

#Person2#: Perfect. We'll review the budget, finalize the project timeline, assign tasks and discuss the launch plan.
```

### Generated Summary

```text
Sarah and her colleague confirmed their project meeting and planned to review the presentation, sales figures, customer feedback, budget, project timeline, task assignments and launch strategy.
```

This example demonstrates how the fine tuned FLAN-T5 model converts a multi turn conversation into a concise and meaningful summary.

---

## Streamlit Application

A lightweight Streamlit application is included for testing the model.

Features:

- Paste any conversation
- Generate summaries instantly
- View results through a simple web interface

---

## Docker

The inference service is containerized using Docker.

### Build Image

```bash
docker build -t flan-t5-api -f inference/Dockerfile .
```

### Run Container

```bash
docker run -p 8080:8080 --env-file .env flan-t5-api
```

---

## CI/CD Pipeline

GitHub Actions automates the deployment workflow.

Whenever changes are pushed to the inference code:

- Checkout the latest source code
- Configure AWS credentials
- Login to Amazon ECR
- Build the Docker image using Docker Buildx
- Tag the image
- Push the image to Amazon ECR

This eliminates the need to manually build and publish Docker images.

---

## Model Logging

Inference requests are logged using **Weights & Biases Weave** to monitor model predictions and keep track of inference activity.

---

## Future Improvements

- Deploy using AWS App Runner
- Model versioning
- Batch inference
- GPU deployment
- Kubernetes deployment
- Automatic retraining pipeline
- Unit and integration testing

---

## Author

**Kshitij Bhanu**
