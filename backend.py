from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Cấu hình
model_file = "C:/Project/models/PhoGPT-4B-Chat-Q4_K_M.gguf"
vector_db_path = "C:/Project/venv/vectorstore/db_history"

# Load LLM
def load_llm(model_path): 
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.01,
        n_gpu_layers=20,
        max_new_tokens=256,
        n_ctx=2048  # nếu model hỗ trợ
    )
    return llm

# Load vector DB và truy vấn BM25
def get_context_from_bm25(question, k=2):
    embedding_model = GPT4AllEmbeddings(model_file=model_file)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

# Tạo prompt cho LLM
def build_prompt(context, question):
    template = """[Vai trò]
                  Bạn là một trợ lý AI thông minh, am hiểu sâu sắc về lịch sử Việt Nam và thế giới, đặc biệt là các giai đoạn lịch sử quan trọng như thời kỳ phong kiến, thời kỳ thuộc địa, chiến tranh và hiện đại.
                  Khi người dùng đặt câu hỏi liên quan đến lịch sử, hãy ưu tiên phân tích đúng ngữ cảnh, nêu rõ nguyên nhân – kết quả, và cung cấp thông tin chính xác dựa trên tài liệu lịch sử chính thống.
                  Nếu người dùng đặt một câu hỏi **không liên quan đến lịch sử**, ví dụ về khoa học, đời sống, hoặc chính bạn, bạn vẫn có thể trả lời một cách tự nhiên, thông minh và thân thiện – miễn là câu hỏi hợp lý và không yêu cầu thông tin vượt giới hạn.
                  Nếu câu hỏi mơ hồ hoặc thiếu thông tin, hãy lịch sự yêu cầu người dùng làm rõ thêm để có thể giúp đỡ tốt hơn.

                Bối cảnh (nếu có):  {context}
                Câu hỏi:  {question}

                [Format Output]

                - Dựa vào thông tin lịch sử chính xác
                - Trình bày dễ hiểu, dành cho người phổ thông
                - Có thể trích dẫn mốc thời gian, nhân vật lịch sử, sự kiện liên quan
                - Nếu không đủ thông tin, hãy nói rõ rằng bạn không chắc chắn hoặc gợi ý người dùng kiểm tra thêm tài liệu

                Trả lời:"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(context=context, question=question)

# Thực thi với câu hỏi và lấy phản hồi từ vector DB hoặc LLM trực tiếp
def handle_question(question):
    context = get_context_from_bm25(question, k=3)
    if context:  # Nếu có dữ liệu trả về từ vector DB
        final_prompt = build_prompt(context, question)
        response = llm.invoke(final_prompt)
    else:
        # Nếu không có dữ liệu trả về từ vector DB, sử dụng LLM để tự trả lời
        prompt = f"Bạn là một trợ lý AI thông minh. Câu hỏi: {question}."
        response = llm.invoke(prompt)  # Sử dụng mô hình để tự tạo câu trả lời
    return response

# Thực thi
def answer_question(llm, question):
    context = get_context_from_bm25(question)
    if context:  # Nếu có dữ liệu trả về từ vector DB
        prompt = build_prompt(context, question)
    else:
        # Nếu không có dữ liệu trả về từ vector DB, trả lời bằng LLM
        prompt = f"Bạn là một trợ lý AI thông minh. Câu hỏi: {question}."
    
    return llm.invoke(prompt)

# Load LLM
llm = load_llm(model_file)

# Thực thi với câu hỏi
question = "Câu hỏi của bạn ở đây"
response = handle_question(question)

# In ra phản hồi
print(response)
