import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add py/ to path so we can import local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from rag_pipeline import RAGPipeline, SimpleVectorStore, EmbeddingManager

def main():
    print('=' * 70)
    print(' Re-ingesting all files using BAAI/bge-m3 Multilingual Embeddings')
    print('=' * 70)

    # 1. Initialize embedding manager and pipeline in offline mode
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    print('Initializing embedding manager (loading BAAI/bge-m3)...')
    pipeline = RAGPipeline(use_local_embeddings=True)

    # 2. Clear old index/embeddings in vector_db/
    print('Clearing old vector database...')
    pipeline.vector_store.clear()

    # 3. Retrieve all ingested chunks from evaluation_data/
    eval_dir = Path(config.EVAL_DIR)
    chunk_files = list(eval_dir.glob('*_chunks.json'))

    if not chunk_files:
        print('Warning: No chunk files found in evaluation_data/. Ingestion skipped.')
        print('Please run app.py or ingestion scripts first to populate evaluation_data.')
        return

    print(f'Found {len(chunk_files)} chunk files to index.')

    # 4. Read chunks and re-embed in batches
    all_chunks = []
    for cf in chunk_files:
        print(f'Loading chunks from {cf.name}...')
        try:
            with open(cf, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            print(f'Error reading {cf.name}: {e}')

    print(f'Total chunks to embed: {len(all_chunks)}')

    if not all_chunks:
        print('No chunks found to embed.')
        return

    # Embed chunks in batches of 100 for safety and progress logging
    batch_size = 100
    total = len(all_chunks)
    t_start = time.time()

    for i in range(0, total, batch_size):
        batch = all_chunks[i:i + batch_size]
        print(f'Embedding batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size} (Chunks {i} to {min(i+batch_size, total)})...')
        pipeline.ingest_new_documents(batch)

    t_elapsed = time.time() - t_start
    print('=' * 70)
    print(f'Re-ingestion successful!')
    print(f'Total Chunks Embedded: {total}')
    print(f'Total Time: {t_elapsed:.2f} s ({t_elapsed/total*1000:.1f} ms/chunk)')
    print(f'Vector DB: {config.VECTOR_DB_DIR}')
    print('=' * 70)

if __name__ == '__main__':
    main()