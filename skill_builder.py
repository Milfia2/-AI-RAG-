#!/usr/bin/env python3
"""
skill_builder.py — 從 RAG 系統自動萃取知識，生成 Agent 可讀的 skill.md

使用方式：
  python skill_builder.py
  python skill_builder.py --output skill.md
  python skill_builder.py --model gpt-oss-20b
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
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
        print("❌ ChromaDB 尚未建立，請先執行：python data_update.py --rebuild")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        print("❌ Collection 不存在，請先執行 data_update.py --rebuild")
        sys.exit(1)


def rag_query(collection, embed_model, query: str, top_k: int = 8) -> list[dict]:
    vec = embed_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": doc,
            "source": meta.get("source_file", ""),
            "similarity": 1 - dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


import time

def call_llm(prompt, model, system="", retries=3):
    import openai

    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_BASE_URL"),
        timeout=120.0,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"     ⚠️  第 {attempt+1} 次失敗，{wait}s 後重試：{e}")
                time.sleep(wait)
            else:
                raise


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] 來源：{c['source']}（相似度 {c['similarity']:.3f}）\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def gather_knowledge(collection, embed_model, model: str) -> dict:
    """
    向 RAG 系統發出多個「全域問題」，萃取知識庫的核心洞察
    """
    GLOBAL_QUESTIONS = {
        "overview": "這個知識庫涵蓋哪些主要法規標準？影像 AI 醫材需要符合哪些法規框架？請提供全面概述。",
        "core_concepts": "影像類 AI 醫材法規中最重要的核心概念有哪些？請列舉並說明 ISO 14971、IEC 62304、SBOM、軟體安全分類等關鍵概念。",
        "key_trends": "目前影像 AI 醫材法規的最新趨勢與發展方向是什麼？包括 TFDA、FDA、EU 的最新要求。",
        "entities": "這個知識庫中提到了哪些重要機構、標準組織、法規文件、技術框架和工具？",
        "methodology": "影像 AI 醫材從開發到上市，軟體生命週期的標準方法論是什麼？最佳實踐有哪些？",
        "gaps": "目前影像 AI 醫材法規有哪些尚未明確規範的灰色地帶或挑戰？",
        "qa_risk": "ISO 14971 在影像 AI 醫材中如何應用？主要的風險評估考量是什麼？",
        "qa_software": "IEC 62304 的軟體安全分類如何決定？Class C 有哪些額外要求？",
        "qa_sbom": "SBOM（軟體物料清單）在影像 AI 醫材法規中的要求與實踐是什麼？",
        "qa_clinical": "TFDA 對影像 AI 醫材的臨床評估有哪些具體要求？",
    }

    results = {}
    total = len(GLOBAL_QUESTIONS)
    for i, (key, question) in enumerate(GLOBAL_QUESTIONS.items(), 1):
        print(f"  [{i}/{total}] 查詢：{question[:50]}...")
        chunks = rag_query(collection, embed_model, question, top_k=6)
        context = build_context(chunks)

        system = """你是醫療器材法規專家，請基於提供的法規文件內容，給出精確、結構化的回答。
只使用文件中的資訊，保留重要英文專有名詞，用繁體中文回答。"""

        prompt = f"""根據以下法規文件內容回答問題。

{context}

問題：{question}

請給出詳細且結構化的回答（300-500字）。"""

        try:
            results[key] = call_llm(prompt, model, system)
        except Exception as e:
            print(f"     ⚠️  LLM 呼叫失敗：{e}")
            results[key] = f"（查詢失敗：{e}）"

    return results


def get_source_list(collection) -> list[str]:
    """取得所有文件來源清單"""
    try:
        all_docs = collection.get(include=["metadatas"])
        sources = {}
        for meta in all_docs["metadatas"]:
            sf = meta.get("source_file", "")
            if sf and sf not in sources:
                sources[sf] = meta.get("chunk_total", 0)
        return [f"{k} ({v} chunks)" for k, v in sorted(sources.items())]
    except Exception:
        return []


def render_skill_md(knowledge: dict, sources: list[str], doc_count: int) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""# Skill: 影像類 AI 醫材法規與軟體生命週期

## Metadata
- **知識領域**：醫療器材法規 / AI 醫材軟體生命週期 / 影像 AI 臨床驗證
- **資料來源數量**：{doc_count} 份文件
- **最後更新時間**：{today}
- **適用 Agent 類型**：醫療器材法規顧問、AI 醫材開發助理、法規查詢機器人

## Overview

{knowledge.get('overview', '')}

## Core Concepts

{knowledge.get('core_concepts', '')}

## Key Trends

{knowledge.get('key_trends', '')}

## Key Entities

{knowledge.get('entities', '')}

## Methodology & Best Practices

{knowledge.get('methodology', '')}

## Knowledge Gaps & Limitations

{knowledge.get('gaps', '')}

## Example Q&A

### Q1: ISO 14971 在影像 AI 醫材中如何應用？
{knowledge.get('qa_risk', '')}

---

### Q2: IEC 62304 的軟體安全分類如何決定？
{knowledge.get('qa_software', '')}

---

### Q3: SBOM 在法規審查中的角色是什麼？
{knowledge.get('qa_sbom', '')}

---

### Q4: TFDA 對影像 AI 醫材的臨床評估要求？
{knowledge.get('qa_clinical', '')}

## Source References

本 Skill 基於以下文件建構：

| 文件 | 說明 |
|---|---|
| ISO14971 | ISO 14971:2019 醫療器材風險管理國際標準 |
| IEC62304 | IEC 62304:2006+AMD1:2015 醫療器材軟體生命週期 |
| TFDA_cybersecurity_guideline | 適用於製造業者之醫療器材網路安全指引（TFDA） |
| TFDA_cybersecurity_SpO2 | 醫療器材網路安全評估分析參考範本（血氧濃度應用軟體） |
| TFDA_cybersecurity_glucose | 醫療器材網路安全評估分析參考範本（葡萄糖試驗系統） |
| TFDA_EP_requirements | 醫療器材安全性與功效性基本規範及技術文件摘要 |
| TFDA_AI_ML_clinical_eval | 人工智慧/機器學習技術之醫療器材軟體臨床評估指導原則 |

### 已索引文件清單
{"".join(f"- {s}" + chr(10) for s in sources) if sources else "（執行 skill_builder.py 後自動填入）"}

### 資料授權聲明
- TFDA 文件：台灣衛生福利部食品藥物管理署公開文件
- ISO/IEC 標準：透過學術機構授權使用
- 自製筆記：個人著作

---
*本 Skill 由 `skill_builder.py` 自動生成，基於 RAG 系統對法規文件庫的系統性知識萃取。*
"""


def main():
    parser = argparse.ArgumentParser(
        description="從 RAG 知識庫自動萃取知識，生成 skill.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python skill_builder.py
  python skill_builder.py --output skill.md --model gpt-oss-20b
        """,
    )
    parser.add_argument("--output", default="skill.md", help="輸出檔案路徑（預設：skill.md）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM 模型（預設：{DEFAULT_MODEL}）")

    args = parser.parse_args()

    print("🏥 Skill Builder — 醫療器材影像 AI 法規知識萃取")
    print("=" * 60)

    print("\n⚙️  載入 Embedding 模型與向量資料庫...")
    embed_model = load_embedding_model()
    collection = get_collection()
    doc_count = collection.count()
    print(f"   ✅ ChromaDB 共 {doc_count:,} 個 chunks")

    print(f"\n🔍 開始知識萃取（模型：{args.model}）...")
    knowledge = gather_knowledge(collection, embed_model, args.model)

    sources = get_source_list(collection)

    print(f"\n✍️  生成 skill.md...")
    content = render_skill_md(knowledge, sources, doc_count)

    output_path = Path(args.output)
    output_path.write_text(content, encoding="utf-8")
    print(f"✅ skill.md 已生成：{output_path}（{len(content):,} 字元）")


if __name__ == "__main__":
    main()
