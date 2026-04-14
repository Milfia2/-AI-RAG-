#!/usr/bin/env python3
"""
rag_query.py — 醫療器材影像 AI 法規 RAG 問答 CLI

使用方式：
  python rag_query.py                          # 互動式模式
  python rag_query.py --query "ISO 14971 的風險管理流程？"
  python rag_query.py --query "..." --top-k 8 --model gpt-oss-20b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION_NAME    = "medical_device_ai_regulations"
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_MODEL      = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_K      = int(os.getenv("DEFAULT_TOP_K", "6"))


def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except ImportError:
        print("❌ 請安裝：pip install sentence-transformers")
        sys.exit(1)


def get_collection():
    try:
        import chromadb
    except ImportError:
        print("❌ 請安裝：pip install chromadb")
        sys.exit(1)

    if not CHROMA_PERSIST_DIR.exists():
        print(f"❌ ChromaDB 尚未建立，請先執行：python data_update.py --rebuild")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' 不存在，請先執行 data_update.py --rebuild")
        sys.exit(1)


def retrieve(collection, embedding_model, query: str, top_k: int) -> list[dict]:
    """向量檢索：Query Embedding → Cosine Similarity Search"""
    query_vec = embedding_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source_file": meta.get("source_file", "unknown"),
            "doc_title": meta.get("doc_title", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total", 0),
            "similarity": 1 - dist,  # cosine distance → similarity
        })
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """組裝 RAG Prompt"""
    system_prompt = """你是一位專精於醫療器材法規的知識助理，特別熟悉：
- ISO 14971（醫療器材風險管理）
- IEC 62304（醫療器材軟體生命週期）
- TFDA 醫療器材網路安全指引
- AI/ML 醫療器材軟體臨床評估規範

請根據提供的法規文件內容回答問題。
規則：
1. 只使用提供的文件內容，不要捏造資訊
2. 每個重要陳述後，用 [來源：文件名稱, Chunk X/Y] 格式標註引用
3. 若文件中找不到答案，明確說明「提供的文件中未涵蓋此內容」
4. 回答使用繁體中文，保留重要的英文專有名詞"""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"【文件 {i}】來源：{chunk['source_file']} "
            f"（Chunk {chunk['chunk_index'] + 1}/{chunk['chunk_total']}，"
            f"相似度：{chunk['similarity']:.3f}）\n{chunk['text']}"
        )

    context = "\n\n" + "─" * 40 + "\n\n".join(context_parts)
    user_prompt = f"根據以下法規文件內容，回答問題：\n\n{context}\n\n{'─' * 40}\n\n問題：{query}"

    return system_prompt, user_prompt


def call_llm(system_prompt, user_prompt, model, conversation_history):
    import openai

    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_BASE_URL"),
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def display_sources(chunks: list[dict]) -> None:
    print("\n📚 引用來源：")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk['source_file']} — Chunk {chunk['chunk_index'] + 1}/{chunk['chunk_total']} "
              f"（相似度：{chunk['similarity']:.3f}）")
        preview = chunk['text'][:120].replace('\n', ' ')
        print(f"      「{preview}...」")


def run_query(query: str, collection, embedding_model, model: str,
              top_k: int, conversation_history: list[dict]) -> str:
    """執行單次 RAG 查詢"""
    chunks = retrieve(collection, embedding_model, query, top_k)
    system_prompt, user_prompt = build_prompt(query, chunks)

    print(f"\n🔍 檢索到 {len(chunks)} 個相關段落，呼叫 {model}...\n")
    answer = call_llm(system_prompt, user_prompt, model, conversation_history)

    print(answer)
    display_sources(chunks)

    return answer


def interactive_mode(collection, embedding_model, model: str, top_k: int) -> None:
    """互動式多輪對話（保留最近 3 輪歷史）"""
    print(f"\n🏥 醫療器材影像 AI 法規 RAG 問答系統")
    print(f"   模型：{model} | Top-K：{top_k} | 輸入 'exit' 結束\n")

    conversation_history: list[dict] = []
    max_history_turns = 3

    while True:
        try:
            query = input("❓ 你的問題：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再見！")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye", "離開", "結束"):
            print("👋 再見！")
            break

        answer = run_query(query, collection, embedding_model, model, top_k, conversation_history)

        # 保留最近 N 輪對話歷史
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > max_history_turns * 2:
            conversation_history = conversation_history[-(max_history_turns * 2):]

        print()


def main():
    parser = argparse.ArgumentParser(
        description="醫療器材影像 AI 法規 RAG 問答 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python rag_query.py                               # 互動式模式
  python rag_query.py --query "IEC 62304 的軟體分類？"
  python rag_query.py --query "..." --top-k 8 --model gpt-oss-20b
        """,
    )
    parser.add_argument("--query", "-q", help="單次查詢問題（不指定則進入互動模式）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"檢索 chunk 數量（預設：{DEFAULT_TOP_K}）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM 模型名稱（預設：{DEFAULT_MODEL}）")

    args = parser.parse_args()

    print("⚙️  載入 Embedding 模型與向量資料庫...")
    embedding_model = load_embedding_model()
    collection = get_collection()
    print(f"   ✅ ChromaDB 共 {collection.count():,} 個 chunks")

    if args.query:
        run_query(args.query, collection, embedding_model, args.model, args.top_k, [])
    else:
        interactive_mode(collection, embedding_model, args.model, args.top_k)


if __name__ == "__main__":
    main()
