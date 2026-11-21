"""Personal Advanced RAG Platform — Streamlit Dashboard.

Run via:
    streamlit run src/personal_rag/ui/app.py
"""

import os
from pathlib import Path
import streamlit as st

from personal_rag.service import PersonalRAGService
from personal_rag.core.config import get_settings
from personal_rag.generation.extractive_synthesizer import ExtractiveSynthesizerProvider
from personal_rag.generation.openai_provider import OpenAICompatibleProvider
from personal_rag.generation.custom_llm_provider import CustomScratchLLMProvider


@st.cache_resource
def load_service():
    """Initialize cached RAG service instance."""
    return PersonalRAGService()


def main():
    st.set_page_config(
        page_title="Personal Advanced RAG Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    service = load_service()
    settings = get_settings()

    # ── Sidebar: Hardware & Knowledge Base Controls ──────────────────────────
    with st.sidebar:
        st.title("🧠 Personal RAG")
        st.caption("Multimodal Research & Intelligence Platform")

        hw_info = service.hw_manager.get_device_info()
        st.divider()
        st.subheader("Hardware & Policy")
        st.success(f"Mode: **{hw_info.device_type.upper()}** (Zero Cost)")
        st.text(f"Device: {hw_info.device_name[:25]}")
        st.caption("GPU credits protected. Cloud autostart: OFF")

        st.divider()
        st.subheader("Generation Engine")
        gen_engine_choice = st.selectbox(
            "Model Backend",
            [
                "Local Extractive (Offline CPU)",
                "Custom Scratch LLM (500M / PyTorch / HF)",
                "Local Ollama (127.0.0.1:11434)",
                "OpenAI / Custom Endpoint"
            ]
        )

        if gen_engine_choice == "Custom Scratch LLM (500M / PyTorch / HF)":
            custom_weights_path = st.text_input("Checkpoint Path (.pt, .bin, or dir)", value="./models/my_500m_model")
            device_choice = st.selectbox("Inference Device", ["cpu", "cuda", "rocm"])
            st.caption("500M parameters runs comfortably on CPU (~1GB RAM, zero GPU cost).")
            if st.button("Load Custom Model"):
                try:
                    service.model_provider = CustomScratchLLMProvider(model_path=custom_weights_path, device=device_choice)
                    service.model_provider.load_model()
                    service.generation_engine.provider = service.model_provider
                    st.success("Custom 500M model loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load custom weights: {e}")
        elif gen_engine_choice == "Local Ollama (127.0.0.1:11434)":
            ollama_model = st.text_input("Ollama Model", value="qwen2.5-vl:7b")
            ollama_url = st.text_input("Endpoint URL", value="http://127.0.0.1:11434/v1")
            service.model_provider = OpenAICompatibleProvider(endpoint=ollama_url, model_name=ollama_model)
            service.generation_engine.provider = service.model_provider
        elif gen_engine_choice == "OpenAI / Custom Endpoint":
            api_endpoint = st.text_input("API Base URL", value="https://api.openai.com/v1")
            api_key = st.text_input("API Key", type="password")
            custom_model = st.text_input("Model Name", value="gpt-4o-mini")
            if api_key:
                service.model_provider = OpenAICompatibleProvider(
                    endpoint=api_endpoint, model_name=custom_model, api_key=api_key
                )
                service.generation_engine.provider = service.model_provider
        else:
            if not isinstance(service.model_provider, ExtractiveSynthesizerProvider):
                service.model_provider = ExtractiveSynthesizerProvider()
                service.generation_engine.provider = service.model_provider

        st.divider()
        st.subheader("Retrieval Controls")
        dense_w = st.slider("Dense Semantic Weight", 0.0, 1.0, 0.65, 0.05)
        lexical_w = st.slider("Lexical (BM25) Weight", 0.0, 1.0, 0.35, 0.05)
        top_k = st.slider("Top-K Candidates", 3, 20, 10, 1)

        st.divider()
        st.subheader("Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, CSV, TXT, MD, Code)",
            accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("Ingest Uploaded Files", type="primary"):
                upload_dir = settings.documents_dir / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                indexed_count = 0
                for f in uploaded_files:
                    target = upload_dir / f.name
                    target.write_bytes(f.getbuffer())
                    rep = service.ingest_path(target)
                    indexed_count += rep.files_indexed
                st.success(f"Successfully processed {len(uploaded_files)} files ({indexed_count} indexed)!")
                st.rerun()

    # ── Main Tabs: Chat, Knowledge Base, User Memory ─────────────────────────
    tab_chat, tab_kb, tab_memory, tab_bench = st.tabs([
        "💬 Research Assistant", "📚 Knowledge Base", "🧠 User Memory", "📊 System Health"
    ])

    # 1. Chat Tab
    with tab_chat:
        st.subheader("Interactive Research Assistant")
        docs_count = len(service.sqlite_store.list_documents(1000))
        st.caption(f"Connected to Knowledge Base: **{docs_count} documents** indexed.")

        if "session_id" not in st.session_state:
            st.session_state.session_id = "default_ui_session"
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    st.caption("📎 **Verified Supporting Sources:**")
                    for cit in msg["citations"]:
                        with st.expander(f"📄 {cit['source_path']} (Page {cit['page_number']})"):
                            st.write(cit["excerpt"])

        # Input query
        if prompt := st.chat_input("Ask a question about your documents, code, or data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents & verifying evidence..."):
                    result = service.answer_query(
                        query=prompt,
                        session_id=st.session_state.session_id,
                        top_k=top_k
                    )

                if result.is_refusal:
                    answer_text = (
                        "⚠️ **Insufficient Evidence:** The indexed documents do not contain enough verified evidence "
                        "to answer this question. (Refusal enforced: zero hallucination policy)."
                    )
                else:
                    answer_text = result.answer

                st.markdown(answer_text)

                cit_payload = []
                if result.citations:
                    st.caption("📎 **Verified Supporting Sources:**")
                    for cit in result.citations:
                        cit_payload.append({
                            "source_path": cit.source_path,
                            "page_number": cit.page_number,
                            "excerpt": cit.excerpt
                        })
                        with st.expander(f"📄 {cit.source_path} (Page {cit.page_number})"):
                            st.write(cit.excerpt)

                st.caption(
                    f"⚡ Strategy: `{result.strategy_used}` | Total Latency: `{result.latency_breakdown_ms.get('total_ms', 0):.1f} ms`"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "citations": cit_payload
                })

    # 2. Knowledge Base Tab
    with tab_kb:
        st.subheader("Registered Documents Catalog")
        docs = service.sqlite_store.list_documents(limit=100)
        if not docs:
            st.info("No documents indexed yet. Upload files in the sidebar or run CLI ingestion.")
        else:
            doc_data = [
                {
                    "Title": d.title,
                    "Path": d.source_path,
                    "Size (KB)": f"{d.file_size_bytes / 1024.0:.1f}",
                    "Pages": d.page_count,
                    "Format": d.mime_type.split("/")[-1].upper(),
                    "Indexed At": d.created_at[:19]
                }
                for d in docs
            ]
            st.dataframe(doc_data, use_container_width=True)

    # 3. User Memory Tab
    with tab_memory:
        st.subheader("Explicit User-Controlled Memory")
        st.caption("Manage persistent personal context, research notes, and model preferences.")

        col_new_k, col_new_v, col_btn = st.columns([2, 4, 1])
        with col_new_k:
            new_key = st.text_input("Memory Key", placeholder="e.g. project_focus")
        with col_new_v:
            new_val = st.text_input("Memory Value", placeholder="e.g. Multimodal RAG with Qwen2.5")
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Save Memory"):
                if new_key and new_val:
                    service.memory_manager.set_memory(new_key, new_val)
                    st.success("Saved!")
                    st.rerun()

        st.divider()
        memories = service.memory_manager.get_all_memories()
        if memories:
            for m in memories:
                col_k, col_v, col_del = st.columns([3, 6, 1])
                col_k.write(f"**{m['key']}**")
                col_v.write(m["value"])
                if col_del.button("🗑️", key=f"del_{m['key']}"):
                    service.memory_manager.remove_memory(m["key"])
                    st.rerun()
        else:
            st.info("No personal memories stored yet.")

    # 4. System Health Tab
    with tab_bench:
        st.subheader("Platform Telemetry & Policy Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Compute Engine", hw_info.device_type.upper(), "CPU Protected")
        col2.metric("Total Documents", len(service.sqlite_store.list_documents(1000)))
        col3.metric("Vector Storage", "Embedded Local (Zero Cost)")

        st.write("### Active Settings")
        st.json({
            "cpu_only": settings.cpu_only,
            "gpu_execution_approved": settings.gpu_execution_approved,
            "embedding_model": settings.embedding_model_name,
            "rrf_k": settings.rrf_k,
            "dense_weight": settings.dense_search_weight,
            "lexical_weight": settings.lexical_search_weight,
            "abstain_threshold": settings.abstain_similarity_threshold,
            "sqlite_db_path": str(settings.sqlite_db_path)
        })


if __name__ == "__main__":
    main()
