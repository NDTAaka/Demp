import streamlit as st
from backend import load_llm, answer_question

st.set_page_config(page_title="Chatbot Lịch Sử", layout="centered")
st.title("🤖 Trợ Lý Lịch Sử Thông Minh")

# Đường dẫn mô hình
model_path = "C:/Project/models/PhoGPT-4B-Chat-Q4_K_M.gguf"

# Load model LLM một lần
if "llm" not in st.session_state:
    with st.spinner("🔄 Đang tải mô hình..."):
        st.session_state.llm = load_llm(model_path)

# Hộp nhập câu hỏi
question = st.text_input("📌 Nhập câu hỏi của bạn (về lịch sử hoặc bất kỳ chủ đề nào):", "")

if question:
    with st.spinner("✍️ Đang suy nghĩ..."):
        response = answer_question(st.session_state.llm, question)
        st.markdown("### 🧠 Trả lời:")
        st.success(response)
