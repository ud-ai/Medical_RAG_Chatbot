"""
Medical RAG Chatbot — Standalone Deployment App
================================================
Deploy on: Hugging Face Spaces | Render | Railway | any Python server

File Structure Expected:
  app.py                  ← this file
  faiss_index/            ← FAISS index folder (generated in Colab notebook)
    index.faiss
    index.pkl
  medquad_2000.csv        ← cleaned dataset (generated in Colab notebook)
  requirements.txt        ← dependencies

Setup:
  1. Run the Colab notebook to generate faiss_index/ and medquad_2000.csv
  2. Download them from Colab: Files panel → right-click → Download
  3. Place them alongside this app.py
  4. Set env variable: GEMINI_API_KEY=your_key
  5. Run: python app.py
"""

import os
import gradio as gr
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

# Load environment variables from .env file if present
from pathlib import Path
try:
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ Loaded environment variables from .env")
        except ImportError:
            print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# ════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "./faiss_index")
DATASET_PATH    = os.environ.get("DATASET_PATH",    "./medquad_2000.csv")
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 4

# ════════════════════════════════════════════════════════
#  RESPONSE CACHING (Reduces quota usage)
# ════════════════════════════════════════════════════════
response_cache = {}  # Store {query_hash: (answer, sources)}

def get_cache_key(user_message):
    """Normalize query for caching (case-insensitive, whitespace-trimmed)"""
    return user_message.strip().lower()

# ════════════════════════════════════════════════════════
#  LOAD EMBEDDINGS + VECTOR STORE
# ════════════════════════════════════════════════════════
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)
print("✅ Embeddings loaded")


def build_or_load_vectorstore():
    """Load existing FAISS index, or build from CSV if index not found."""
    if os.path.exists(FAISS_INDEX_DIR):
        print(f"📂 Loading FAISS index from: {FAISS_INDEX_DIR}")
        vs = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS loaded — {vs.index.ntotal} vectors")
        return vs

    # Fallback: build from CSV
    print(f"⚠️  FAISS index not found. Building from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Neither {FAISS_INDEX_DIR}/ nor {DATASET_PATH} found. "
            "Please run the Colab notebook first to generate these files."
        )

    df = pd.read_csv(DATASET_PATH)
    docs = [
        Document(
            page_content=f"Question: {row['question']}\nAnswer: {row['answer']}",
            metadata={"index": i, "question": row["question"]}
        )
        for i, row in df.iterrows()
    ]
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_DIR)
    print(f"✅ FAISS index built & saved — {vs.index.ntotal} vectors")
    return vs


vectorstore = build_or_load_vectorstore()

# ════════════════════════════════════════════════════════
#  MEDICAL PROMPT TEMPLATE
# ════════════════════════════════════════════════════════
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a highly knowledgeable and empathetic medical assistant. \
Use ONLY the provided context from the MedQuAD medical database to answer the question. \
If the context does not contain enough information, say so clearly and recommend consulting a healthcare professional.

Guidelines:
- Be precise, factual, and medically accurate
- Use simple language when possible
- Always remind users to consult a doctor for personal medical advice
- Structure your answer clearly if it is lengthy

Context from MedQuAD:
{context}

Patient Question: {question}

Medical Answer:"""
)

# ════════════════════════════════════════════════════════
#  LLM INITIALIZATION WITH FALLBACK
# ════════════════════════════════════════════════════════

# Model fallback chain
AVAILABLE_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
]

# Global state for current LLM
current_llm = None
current_model = None
model_index = 0

def init_llm(model_name):
    """Initialize LLM with the given model name."""
    global current_llm, current_model
    try:
        current_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
            max_output_tokens=1024
        )
        current_model = model_name
        print(f"✅ LLM initialized: {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to initialize {model_name}: {e}")
        return False

def switch_to_next_model():
    """Switch to the next available model when rate limited."""
    global model_index, current_llm, current_model, qa_chain
    
    model_index += 1
    if model_index >= len(AVAILABLE_MODELS):
        print("⚠️ All models exhausted!")
        return False
    
    model_name = AVAILABLE_MODELS[model_index]
    print(f"\n🔄 Switching model from {current_model} → {model_name}")
    
    if init_llm(model_name):
        # Rebuild the chain with new LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=current_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": MEDICAL_PROMPT},
            return_source_documents=True,
        )
        print(f"✅ Switched to {model_name}")
        return True
    return False

def manual_switch_model(selected_model):
    """Manually switch to a user-selected model."""
    global model_index, current_llm, current_model, qa_chain
    
    if selected_model == current_model:
        return f"Already using {current_model}"
    
    # Find the index of selected model
    try:
        model_index = AVAILABLE_MODELS.index(selected_model)
    except ValueError:
        return f"Model {selected_model} not found"
    
    print(f"\n👤 User switching model: {current_model} → {selected_model}")
    
    if init_llm(selected_model):
        # Rebuild the chain with new LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=current_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": MEDICAL_PROMPT},
            return_source_documents=True,
        )
        print(f"✅ Manually switched to {selected_model}")
        return f"Successfully switched to {selected_model}"
    return f"Failed to switch to {selected_model}"


# Initialize with first model
print("⏳ Initializing Gemini LLM...")
for model in AVAILABLE_MODELS:
    if init_llm(model):
        break
else:
    raise RuntimeError(f"Failed to initialize any Gemini model")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=current_llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": MEDICAL_PROMPT},
    return_source_documents=True,
)

print("✅ RAG chain ready")


# ════════════════════════════════════════════════════════
#  CHAT FUNCTION (Gradio 6.0 messages format)
# ════════════════════════════════════════════════════════
def medical_chat(user_message: str, history: list):
    global qa_chain
    
    if not user_message.strip():
        return history

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        error_msg = "⚠️ GEMINI_API_KEY is not set. Please configure it as an environment variable."
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": error_msg})
        return history

    answer = None
    exc = None
    
    # Check response cache first
    cache_key = get_cache_key(user_message)
    if cache_key in response_cache:
        print(f"✅ Cache hit: {cache_key[:40]}...")
        answer, sources = response_cache[cache_key]
    else:
        try:
            result  = qa_chain.invoke({"query": user_message})
            answer  = result["result"]
            sources = result.get("source_documents", [])
            # Cache the successful response
            response_cache[cache_key] = (answer, sources)
        except Exception as outer_exc:
            answer = None
            exc = outer_exc
    
    # Format answer with sources if available
    if answer is not None:
        sources = response_cache[cache_key][1] if cache_key in response_cache else []
        if sources:
            refs = "\n".join(
                f"- {doc.metadata.get('question', '')[:70]}..."
                for doc in sources[:2]
            )
            answer += f"\n\n---\n📚 *Sources retrieved from MedQuAD:*\n{refs}"
    
    # Handle errors if API call failed
    if answer is None and exc is not None:
        error_str = str(exc)
        # Check for quota exceeded error
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            print(f"\n🔴 Rate limit hit for {current_model}!")
            
            # Try to switch to next model
            if switch_to_next_model():
                # Retry with the new model
                try:
                    result = qa_chain.invoke({"query": user_message})
                    answer = result["result"]
                    sources = result.get("source_documents", [])
                    
                    if sources:
                        refs = "\n".join(
                            f"- {doc.metadata.get('question', '')[:70]}..."
                            for doc in sources[:2]
                        )
                        answer += f"\n\n---\n📚 *Sources retrieved from MedQuAD:*\n{refs}"
                    
                    answer = f"🔄 **Model switched to {current_model}** (previous model quota exceeded)\n\n{answer}"
                except Exception as retry_exc:
                    answer = f"⚠️ Error with {current_model}: {retry_exc}"
            else:
                # Show available models to try
                remaining_models = [m for i, m in enumerate(AVAILABLE_MODELS) if i > model_index]
                models_list = "\n".join([f"   - {m}" for m in remaining_models]) if remaining_models else "   (None available)"
                
                answer = (
                    "⚠️ **Rate Limit Reached**\n\n"
                    f"Current model **{current_model}** has hit its quota limit.\n\n"
                    "**Options:**\n"
                    "1. 🔄 **Try a different model** using the dropdown above\n"
                    f"   Available models:\n{models_list}\n\n"
                    "2. 📅 Wait until tomorrow (quota resets daily)\n"
                    "3. 💳 **Upgrade to paid** ([Enable billing](https://console.cloud.google.com/billing))\n"
                    "4. 🔑 Use a different API key\n\n"
                    "💡 **Tip:** Use the model selector above to manually switch!"
                )
        else:
            answer = f"⚠️ Error: {error_str}\n\nCheck your API key and network connection."

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return history


# ════════════════════════════════════════════════════════
#  GRADIO UI
# ════════════════════════════════════════════════════════
SAMPLE_QS = [
    "🩺 What are the symptoms of type 2 diabetes?",
    "❤️  How is hypertension diagnosed and treated?",
    "🫁 What triggers an asthma attack?",
    "🧠 What is Alzheimer's disease?",
    "💊 What are common side effects of chemotherapy?",
]

CSS = """
#chatbot { min-height: 460px; }
.disclaimer-box {
    background: #FFFFFF; border-left: 4px solid #EAB308;
    padding: 10px 16px; border-radius: 6px; font-size: 14px;
    margin-bottom: 10px; color: #000000 !important;
}
.disclaimer-box, .disclaimer-box * {
    color: #000000 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #000000 !important;
}
.disclaimer-box strong { font-weight: 700; }
.disclaimer-box em { font-style: italic; font-weight: 500; }
footer { display: none !important; }
"""

with gr.Blocks(title="Medical RAG Chatbot") as demo:

    gr.HTML("""
    <div style='text-align:center; padding:18px 0 8px'>
      <h1 style='font-size:1.9em; margin:0'>🏥 Medical RAG Chatbot</h1>
      <p style='color:#6B7280; margin:4px 0 0'>
        FAISS &nbsp;·&nbsp; MedQuAD 2,000 Q&amp;A pairs
      </p>
    </div>
    """)

    gr.HTML("""
        <div class='disclaimer-box' style='background:#FFFFFF; border-left:4px solid #EAB308; padding:10px 16px; border-radius:6px; font-size:14px; margin-bottom:10px; color:#000000 !important;'>
            <strong style='color:#000000; font-weight:700;'>⚠️ Disclaimer:</strong> <span style='color:#000000;'>This chatbot provides <em style='color:#000000;'>general medical information only</em>.
      It is <strong style='color:#000000;'>not</strong> a substitute for professional medical advice, diagnosis, or treatment.
      Always consult a qualified healthcare provider for personal health concerns.</span>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Medical Assistant"
            )
            msg_box = gr.Textbox(
                placeholder="Ask a medical question...",
                label="Your Question",
                container=False
            )
            with gr.Row():
                send_btn = gr.Button("Send 💬", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=current_model,
                    label="⚡ Select Gemini Model",
                    interactive=True
                )
                switch_btn = gr.Button("Switch Model", variant="secondary", size="sm")
            
            model_status = gr.Textbox(label="Model Status", value=f"Currently using: {current_model}", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Try These")
            for q in SAMPLE_QS:
                gr.Button(q, variant="secondary").click(
                    fn=lambda x=q: x, outputs=msg_box
                )

    with gr.Row():
        gr.Markdown("📦 **Dataset:** MedQuAD 2K")
        gr.Markdown("🧠 **Embed:** MiniLM-L6-v2")
        gr.Markdown("🔍 **VectorDB:** FAISS")

    def handle_model_switch(selected_model):
        result = manual_switch_model(selected_model)
        return f"✅ Switched to: {current_model}"
    
    switch_btn.click(
        fn=handle_model_switch,
        inputs=[model_selector],
        outputs=[model_status]
    )

    send_btn.click(
        fn=medical_chat,
        inputs=[msg_box, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        [msg_box]
    )
    msg_box.submit(
        fn=medical_chat,
        inputs=[msg_box, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        [msg_box]
    )
    clear_btn.click(lambda: [], None, [chatbot])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7865)),
        share=False,  # keep false for cloud deployments
        debug=False,
        css=CSS,
        theme=gr.themes.Soft()
    )
