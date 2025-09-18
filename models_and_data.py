import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import evaluate

# ---------------------------
# Load Hugging Face Token
# ---------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in .env file")

# ---------------------------
# Dataset
# ---------------------------
dataset = load_dataset("squad", split="validation[:50]")

# ---------------------------
# Metric
# ---------------------------
metric = evaluate.load("squad")

# ---------------------------
# Models dictionary
# ---------------------------
MODELS = {
    "distilbert": "distilbert-base-cased-distilled-squad",
    "BERT": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa": "deepset/roberta-base-squad2"
}

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate_model(model_name, dataset=dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

    preds, refs = [], []
    for example in dataset:
        prediction = qa(question=example["question"], context=example["context"])
        preds.append({"id": example["id"], "prediction_text": prediction["answer"]})
        refs.append({"id": example["id"], "answers": example["answers"]})

    return metric.compute(predictions=preds, references=refs)

if __name__ == "__main__":
    # Example usage
    context = """The Nile River is one of the longest rivers in the world.
    It flows through Egypt, Sudan, and Ethiopia. It is the lifeline
    of Sudan's economy, supporting agriculture and industry."""

    question = "Which countries does the Nile River flow through?"

    model_name = MODELS["BERT"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    result = qa_pipeline(question=question, context=context)
    print("Answer:", result['answer'])
