import streamlit as st
from transformers import pipeline

MODELS = {
    "distilbert": "distilbert-base-cased-distilled-squad",
    "BERT": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa": "deepset/roberta-base-squad2",
}

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("🧠 Multi-Model QA System")
selected_model = st.sidebar.radio(
    "Select a model to use:",
    options=list(MODELS.keys()),
    index=0
)

# Main UI
st.title("📘 Question Answering with Transformers")
st.markdown("Enter your **context** once, then ask multiple questions. The answers will appear below like a chat.")

context = st.text_area("Context:", height=200, placeholder="Enter your paragraph or text here...")

question = st.text_input("Question:", placeholder="Enter your question here...")

# Load the selected model once
@st.cache_resource
def load_qa_pipeline(model_name):
    return pipeline("question-answering", model=MODELS[model_name], tokenizer=MODELS[model_name])

qa_pipeline = load_qa_pipeline(selected_model)

if st.button("🚀 Ask"):
    if not context or not question:
        st.warning("Please enter both context and question!")
    else:
        result = qa_pipeline(question=question, context=context)
        st.session_state.chat_history.append({
            "question": question,
            "answer": result['answer'],
            "confidence": result['score'],
            "model": selected_model
        })

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"""
    <div style='background-color:#1e293b; color:#f1f5f9; padding:15px; border-radius:10px; 
                border:2px solid #0ea5e9; margin:10px 0;'>
        <h4 style='color:#38bdf8;'>🤖 {chat['model']}</h4>
        <p><strong>Q:</strong> {chat['question']}</p>
        <p><strong>A:</strong> 
            <span style='background-color:#0ea5e9; color:white; padding:5px 10px; border-radius:5px;'>
                {chat['answer']}
            </span>
        </p>
        <p><strong>Confidence:</strong> {chat['confidence']:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
