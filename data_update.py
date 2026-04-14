#!/usr/bin/env python3
"""
data_update.py — 醫療器材影像 AI 法規知識庫 RAG 資料管線

主題：影像類 AI 醫材（含軟體生命週期）法規整理
支援格式：PDF、Markdown、TXT
Vector DB：ChromaDB（純 Python，無需 Docker）
Embedding：sentence-transformers（本地，完全免費）

使用方式：
  python data_update.py                      # 增量更新
  python data_update.py --rebuild            # 全量重建
  python data_update.py --chunk-size 800     # 自訂 chunk 大小
  python data_update.py --dry-run            # 只列出文件，不執行
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── 常數設定 ──────────────────────────────────────────────────────────────────
RAW_DIR        = Path("data/raw")
PROCESSED_DIR  = Path("data/processed")
CHROMA_DIR     = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION_NAME = "medical_device_ai_regulations"
HASH_STORE     = Path(".data_hashes.json")

EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "150"))

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt", ".text"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def _process_worker(args: tuple) -> tuple[str, bool]:
    """
    多進程 worker：解析 + 清理 + chunking（不含 embedding）
    embedding 仍在主進程用 GPU 執行
    """
    raw_path, chunk_size, overlap, force, cur_hash = args
    raw_path = Path(raw_path)

    raw = parse_file(raw_path)
    if raw is None:
        return str(raw_path), False

    cleaned = clean_text(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc = PROCESSED_DIR / (raw_path.stem + ".txt")
    proc.write_text(cleaned, encoding="utf-8")

    chunks = chunk_text(cleaned, chunk_size, overlap)
    return str(raw_path), chunks

# ── Hash（增量更新） ──────────────────────────────────────────────────────────

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes() -> dict:
    if HASH_STORE.exists():
        return json.loads(HASH_STORE.read_text(encoding="utf-8"))
    return {}


def save_hashes(store: dict) -> None:
    HASH_STORE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 文件解析 ──────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> str:
    """PDF 解析：優先 pdfplumber（保留表格），退而 pypdf"""
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                t = page.extract_text()
                if t:
                    parts.append(f"[Page {i}]\n{t}")
        return "\n\n".join(parts)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        for i, page in enumerate(reader.pages, 1):
            t = page.extract_text()
            if t:
                parts.append(f"[Page {i}]\n{t}")
        return "\n\n".join(parts)
    except ImportError:
        raise ImportError("請安裝 PDF 解析套件：pip install pdfplumber 或 pip install pypdf")

def parse_pdf_streaming(path: Path, pages_per_batch: int = 20) -> str:
    """分批讀取 PDF，避免大檔案 OOM"""
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for i in range(0, len(pdf.pages), pages_per_batch):
            batch = pdf.pages[i:i + pages_per_batch]
            for j, page in enumerate(batch, i + 1):
                t = page.extract_text()
                if t:
                    parts.append(f"[Page {j}]\n{t}")
    return "\n\n".join(parts)

def parse_md(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return text


def parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# 解析器對應表（供外部模組 import）
PARSERS = {
    ".pdf": parse_pdf_streaming,
    ".md": parse_md, ".markdown": parse_md,
    ".txt": parse_txt, ".text": parse_txt,
}


def parse_file(path: Path) -> Optional[str]:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        log.warning(f"  ⚠️  不支援格式，略過：{path.name}")
        return None
    try:
        return parser(path)
    except Exception as e:
        log.error(f"  ❌ 解析失敗 {path.name}: {e}")
        return None


# ── 文字清理 ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    移除法規 PDF 常見雜訊：
    - 頁碼模式（第 N 頁 共 M 頁、- N -、Page N of M）
    - ISO/IEC watermark 行
    - 表格框線字符
    - 多餘空白行
    """
    text = re.sub(r"第\s*\d+\s*頁\s*[共/]\s*\d+\s*頁", "", text)
    text = re.sub(r"[-–]\s*\d+\s*[-–]", "", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(ISO|IEC)\s+\d{4,5}[:\-]\d+.*?\n", "\n", text)
    text = re.sub(r"[┌┐└┘├┤┬┴┼─│]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"^[\s\.\-_=*#]+$", "", text, flags=re.MULTILINE)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    法規文件 Chunking 策略（兩層）：
    1. 段落優先切分：在雙換行處切分，保留條文完整性
    2. 超長段落退化為固定長度 + overlap 切分

    設計理由：
    - 法規「條文」是最小語意單位，切斷條文會破壞引用準確性
    - chunk_size=800 適合台灣法規條文平均長度（約 300-600 字）
    - overlap=150 確保跨 chunk 上下文連貫（如條文與其解釋說明）
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                sub = _sliding_chunk(para, chunk_size, overlap)
                chunks.extend(sub[:-1])
                current = sub[-1] if sub else ""
            else:
                current = para

    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 50]


def _sliding_chunk(text: str, size: int, overlap: int) -> list[str]:
    """固定長度 + overlap，並嘗試在語句邊界切分（已修復無窮迴圈 Bug）"""
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size]
        
        # 【關鍵修復 1】如果這已經是最後一塊（包含到底了），直接加入陣列並結束迴圈
        if start + size >= len(text):
            chunks.append(chunk.strip())
            break
            
        for sep in ["。", "！", "？", ".", "!", "?"]:
            last = chunk.rfind(sep)
            if last > size // 2:
                chunk = chunk[:last + 1]
                break
                
        chunks.append(chunk.strip())
        
        # 計算下一個起點的移動步數
        step = len(chunk) - overlap
        
        # 【關鍵修復 2】防止移動步數 <= 0 導致原地踏步或倒退嚕
        if step <= 0:
            step = 1  # 強制至少往前進 1 步
            
        start += step
        
    return [c for c in chunks if c]


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embed_model():
    """載入 sentence-transformers 本地 Embedding 模型（完全免費）"""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"  📥 載入 Embedding 模型：{EMBEDDING_MODEL}")
        log.info("     （首次執行將自動下載，約 420MB）")
        return SentenceTransformer(EMBEDDING_MODEL, device=device)
    except ImportError:
        log.error("❌ 請安裝：pip install sentence-transformers")
        sys.exit(1)

def embed_chunks(model, chunks, batch_size=32):
    all_vecs = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True)
        all_vecs.extend(vecs.tolist())
        # 每批結束後 batch 就被 GC 回收
    return all_vecs


# ── Vector DB（ChromaDB） ─────────────────────────────────────────────────────

def get_collection(rebuild: bool = False):
    """取得或建立 ChromaDB collection"""
    try:
        import chromadb
    except ImportError:
        log.error("❌ 請安裝：pip install chromadb")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info(f"  🗑️  已清空 collection")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert(collection, chunks, embeddings, source_file, doc_title):
    """幂等 upsert：存在則更新，不存在則新增"""
    # 先刪除此文件的舊 chunks
    try:
        old = collection.get(where={"source_file": source_file}, include=["metadatas"])
        if old["ids"]:
            collection.delete(ids=old["ids"])
            log.info(f"     🗑️  刪除舊 chunks：{len(old['ids'])} 筆")
    except Exception:
        pass

    ids = [f"{source_file}::chunk_{i}" for i in range(len(chunks))]
    metas = [
        {
            "source_file": source_file,
            "doc_title": doc_title,
            "chunk_index": i,
            "chunk_total": len(chunks),
            "char_count": len(c),
        }
        for i, c in enumerate(chunks)
    ]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)


# ── 單文件處理 ────────────────────────────────────────────────────────────────

def load_and_process(
    raw_path: Path, model, collection, hashes: dict, force: bool = False
) -> bool:
    """
    處理單一文件：解析 → 清理 → 儲存 processed/ → chunking → embedding → 寫入 DB
    return True 若實際更新
    """
    key = str(raw_path)
    cur_hash = md5(raw_path)

    if not force and hashes.get(key) == cur_hash:
        log.info(f"  ⏭️  未變更：{raw_path.name}")
        return False

    log.info(f"\n  📄 處理：{raw_path.name}")

    raw = parse_file(raw_path)
    if raw is None:
        return False

    cleaned = clean_text(raw)
    log.info(f"     清理後：{len(cleaned):,} 字元")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc = PROCESSED_DIR / (raw_path.stem + ".txt")
    proc.write_text(cleaned, encoding="utf-8")
    log.info(f"     ✅ 儲存至 {proc}")

    chunks = chunk_text(cleaned)
    log.info(f"     📦 {len(chunks)} chunks（size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

    if not chunks:
        log.warning("     ⚠️  無有效 chunk，略過")
        return False

    embeddings = embed_chunks(model, chunks)
    doc_title = raw_path.stem.replace("_", " ").replace("-", " ")
    upsert(collection, chunks, embeddings, raw_path.stem, doc_title)
    log.info(f"     ✅ 寫入 ChromaDB：{len(chunks)} 筆")

    hashes[key] = cur_hash
    return True


# ── 主管線 ────────────────────────────────────────────────────────────────────

def build_pipeline(rebuild=False, files=None):
    """
    主執行流程（冪等設計）
    rebuild=True  → 清空 processed/ 與 ChromaDB，全量重建
    rebuild=False → 僅處理有變動的文件（以 MD5 hash 判斷）
    """
    log.info("=" * 60)
    log.info("🏥 醫療器材影像 AI 法規 RAG — 資料更新管線")
    log.info("=" * 60)

    if not RAW_DIR.exists():
        log.error(f"❌ 找不到 {RAW_DIR}，請先建立並放入原始文件")
        sys.exit(1)

    if files:
        raw_files = [Path(f) for f in files if Path(f).suffix.lower() in SUPPORTED_EXTS]
    else:
        raw_files = [p for p in RAW_DIR.rglob("*")
                     if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    log.info(f"\n📁 找到 {len(raw_files)} 個文件（{', '.join(SUPPORTED_EXTS)}）")

    if not raw_files:
        log.warning("⚠️  data/raw/ 中無支援的文件")
        return

    if rebuild:
        log.info("\n🔄 全量重建模式")
        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)
            log.info(f"   🗑️  清空 {PROCESSED_DIR}")
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            log.info(f"   🗑️  清空 ChromaDB ({CHROMA_DIR})")
        if HASH_STORE.exists():
            HASH_STORE.unlink()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"\n🤖 Embedding：{EMBEDDING_MODEL} (sentence-transformers, 本地)")
    model = get_embed_model()
    log.info(f"\n💾 ChromaDB：{CHROMA_DIR}")
    collection = get_collection(rebuild=rebuild)
    hashes = load_hashes()

    # 過濾出需要處理的文件
    to_process = []
    for path in sorted(raw_files):
        key = str(path)
        cur_hash = md5(path)
        if not rebuild and hashes.get(key) == cur_hash:
            log.info(f"  ⏭️  未變更：{path.name}")
            continue
        to_process.append((str(path), CHUNK_SIZE, CHUNK_OVERLAP, rebuild, cur_hash))

    if not to_process:
        log.info("✅ 所有文件皆為最新，無需更新")
        return

    # 多進程：PDF 解析 + 清理 + chunking
    n_workers = min(os.cpu_count(), len(to_process))
    log.info(f"\n⚡ 多進程處理：{len(to_process)} 份文件，{n_workers} 個 worker")

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_worker, args): args[0]
                   for args in to_process}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="解析+Chunking", unit="份"):
            path_str, chunks = future.result()
            if chunks:
                results[path_str] = chunks

    # 主進程循序：embedding（GPU）+ 寫入 ChromaDB
    log.info(f"\n🔢 Embedding + 寫入（{len(results)} 份文件）")
    for path_str, chunks in tqdm(results.items(), desc="Embedding+寫入", unit="份"):
        path = Path(path_str)
        embeddings = embed_chunks(model, chunks)
        doc_title = path.stem.replace("_", " ")
        upsert(collection, chunks, embeddings, path.stem, doc_title)
        hashes[str(path)] = md5(path)

    save_hashes(hashes)

    log.info("\n" + "=" * 60)
    log.info("📊 完成：")
    log.info(f"   ChromaDB 總 chunks：{collection.count():,}")
    log.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="醫療器材影像 AI 法規知識庫 — 資料收集、清理、Chunking 與向量索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python data_update.py             # 增量更新
  python data_update.py --rebuild   # 全量重建
  python data_update.py --chunk-size 800 --chunk-overlap 150
        """,
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="清空並全量重建索引")
    parser.add_argument("--files", nargs="+", metavar="PATH",
                        help="只更新指定文件（不指定則處理 data/raw/ 全部）")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Chunk 字元數（預設 {CHUNK_SIZE}）")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                        help=f"Overlap 字元數（預設 {CHUNK_OVERLAP}）")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL,
                        help=f"sentence-transformers 模型（預設：{EMBEDDING_MODEL}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="僅列出待處理文件，不實際執行")

    args = parser.parse_args()

    # Override module-level constants with CLI args
    import data_update as _self
    _self.CHUNK_SIZE    = args.chunk_size
    _self.CHUNK_OVERLAP = args.chunk_overlap
    _self.EMBEDDING_MODEL = args.embedding_model

    if args.dry_run:
        files = [p for p in RAW_DIR.rglob("*")
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS] if RAW_DIR.exists() else []
        log.info(f"📁 [Dry Run] 將處理 {len(files)} 個文件：")
        for f in sorted(files):
            log.info(f"   {f}")
        return

    build_pipeline(rebuild=args.rebuild, files=args.files)


if __name__ == "__main__":
    main()
