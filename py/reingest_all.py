"""
reingest_all.py
───────────────
Clears the existing duplicate-laden index & vector store and freshly ingests
every supported document found in the knowledge_corpus directory.

Run from the py/ directory:
    python reingest_all.py
"""

import sys
import json
import shutil
from pathlib import Path

# ── Add py/ to path so we can import local modules ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ingestion import DocumentIngester
from rag_pipeline import RAGPipeline, SimpleVectorStore

SUPPORTED_EXTS = {".txt", ".pdf", ".docx", ".md"}

def clear_stale_data():
    """Remove old evaluation_data chunks + index and vector_db files."""
    print("\n   Clearing stale index and vector store ...")

    # Remove all chunk JSON files and the document index
    eval_dir = Path(config.EVAL_DIR)
    if eval_dir.exists():
        for f in eval_dir.iterdir():
            if f.suffix == ".json":
                f.unlink()
                print(f"   Deleted: {f.name}")

    # Remove vector store files
    vdb_dir = Path(config.VECTOR_DB_DIR)
    if vdb_dir.exists():
        for f in vdb_dir.iterdir():
            f.unlink()
            print(f"   Deleted: {f.name}")

    print("OK Stale data cleared.\n")


def ingest_all_documents():
    """Ingest every supported file from the knowledge_corpus directory."""
    corpus_dir = Path(config.CORPUS_DIR)
    if not corpus_dir.exists():
        print(f"ERROR Knowledge corpus directory not found: {corpus_dir}")
        sys.exit(1)

    files = [f for f in corpus_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        print(f"ERROR No supported documents found in {corpus_dir}")
        sys.exit(1)

    print(f"Found {len(files)} document(s) in knowledge_corpus:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"   * {f.name}  ({size_kb:.1f} KB)")

    # Build fresh ingester and pipeline
    ingester = DocumentIngester()
    pipeline = RAGPipeline()          # Uses API key from env if available

    ingestion_summary = []

    for file_path in sorted(files):
        print(f"\nIngesting: {file_path.name} ...")
        try:
            doc_id, doc_meta = ingester.ingest_document(
                file_path,
                domain="Traditional Knowledge / Sustainable Agriculture",
                chunk_size=800,
                chunk_overlap=150
            )
            # Load chunks and push into vector store
            chunks_file = Path(config.EVAL_DIR) / doc_meta["chunks_file"]
            with open(chunks_file, "r", encoding="utf-8") as fh:
                chunks = json.load(fh)
            pipeline.ingest_new_documents(chunks)

            ingestion_summary.append({
                "file": file_path.name,
                "doc_id": doc_id,
                "chunks": doc_meta["total_chunks"],
                "chars": doc_meta["total_characters"],
                "status": "OK"
            })
        except Exception as exc:
            print(f"   WARNING Failed to ingest {file_path.name}: {exc}")
            ingestion_summary.append({
                "file": file_path.name,
                "status": f"ERROR: {exc}"
            })

    # Print summary table
    print("\n" + "-" * 70)
    print(f"{'File':<45} {'Chunks':>7} {'Status'}")
    print("-" * 70)
    for s in ingestion_summary:
        chunks_str = str(s.get("chunks", "-")).rjust(7)
        print(f"{s['file']:<45} {chunks_str}   {s['status']}")
    print("-" * 70)

    total_chunks = sum(s.get("chunks", 0) for s in ingestion_summary)
    ok_count     = sum(1 for s in ingestion_summary if "OK" in s["status"])
    print(f"\nIngestion complete: {ok_count}/{len(files)} documents, {total_chunks} total chunks.")
    print(f"   Index  -> {config.EVAL_DIR}")
    print(f"   Vector -> {config.VECTOR_DB_DIR}\n")


if __name__ == "__main__":
    print("=" * 70)
    print(" Digital Knowledge Extinction Framework - Full Corpus Re-ingestion")
    print("=" * 70)
    clear_stale_data()
    ingest_all_documents()
