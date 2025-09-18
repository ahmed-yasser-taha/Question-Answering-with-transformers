# 🧠 Multi-Model Question Answering System

This project is a **Question Answering (QA) web application** built with **Streamlit** and **Hugging Face Transformers**. It allows users to input a text paragraph (context) and ask multiple questions about it. The system provides answers using several pre-trained transformer models.

---

## Features

- **Multi-model support:** Choose from DistilBERT, BERT, RoBERTa, or ALBERT.
- **Chat-like history:** Ask multiple questions and see all answers in a conversation-style layout.
- **Confidence scores:** Each answer comes with a model confidence percentage.
- **Interactive UI:** Clean and responsive Streamlit interface.

---

## How it Works

1. **Load a pre-trained model:** The selected transformer model is loaded using Hugging Face pipelines.
2. **Input context and questions:** Users enter a text paragraph and ask questions.
3. **Generate answers:** The model predicts the answer span in the context.
4. **Display results:** Answers and confidence scores are displayed in a chat-style interface.

---

## Supported Models

| Model      | Hugging Face Model ID |
|-----------|----------------------|
| DistilBERT | `distilbert-base-cased-distilled-squad` |
| BERT       | `bert-large-uncased-whole-word-masking-finetuned-squad` |
| RoBERTa    | `deepset/roberta-base-squad2` |


---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/multi-model-qa.git
cd multi-model-qa
