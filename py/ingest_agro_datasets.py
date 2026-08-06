
import sys, json, random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from rag_pipeline import RAGPipeline

AGRO_DIR = Path(config.CORPUS_DIR) / 'Agro dataset'
EVAL_DIR = Path(config.EVAL_DIR)
AGRO_DOMAIN = 'Agro Dataset / Scientific Crop Knowledge'

RAG_READY_FILES = [
    'crop_dataset_rag_ready.json',
    'crop_disease_rag_ready.json',
    'fertilizer_dataset_rag_ready.json',
]
LARGE_RAG_READY_FILES = {
    'crop_yield_dataset_rag_ready.json': 500,
    'soil_dataset_rag_ready.json': 500,
    'weather_dataset_rag_ready.json': 200,
}

def load_rag_ready(filepath, max_records=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        records = json.load(f)
    total = len(records)
    if max_records and total > max_records:
        random.seed(42)
        records = random.sample(records, max_records)
        print(f'  Sampled {max_records}/{total} from {filepath.name}')
    chunks = []
    for i, rec in enumerate(records):
        text = rec.get('text', '').strip()
        if not text:
            continue
        meta = dict(rec.get('metadata', {}))
        meta.setdefault('source', filepath.name)
        meta.setdefault('page', 1)
        meta['domain'] = AGRO_DOMAIN
        meta['char_count'] = len(text)
        meta['chunk_index'] = i
        chunks.append({
            'chunk_id': f'agro_{filepath.stem}_c{i}',
            'text': text,
            'metadata': meta
        })
    return chunks

def preprocess_rice(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = []
    idx = 0
    for rec in data.get('season_data', []):
        dist = ', '.join(rec.get('districts', []))
        nm = rec.get('season_name', '')
        mo = rec.get('sowing_month', '')
        du = rec.get('duration_days', '')
        sr = rec.get('source', 'TNAU')
        t = 'Rice season ' + nm + ' sown ' + mo + ' duration ' + str(du) + ' days. Districts: ' + dist + '. Source: ' + sr + '.'
        chunks.append({
            'chunk_id': f'agro_rice_s_c{idx}',
            'text': t,
            'metadata': {'source': filepath.name, 'page': 1, 'domain': AGRO_DOMAIN,
                        'type': 'rice_season', 'char_count': len(t), 'chunk_index': idx}
        })
        idx += 1
    for rec in data.get('variety_data', []):
        dur = rec.get('duration', '')
        if isinstance(dur, dict):
            dur_str = str(dur.get('min', '?')) + '-' + str(dur.get('max', '?')) + ' days'
        else:
            dur_str = str(dur)
        traits = '; '.join(str(k) + ':' + str(v) for k, v in rec.get('traits', {}).items())
        seasons = '; '.join(rec.get('suitable_seasons', []))
        vn = rec.get('variety_name', '')
        vt = rec.get('variety_type', '')
        sr = rec.get('source', 'TNAU')
        t = 'Rice variety ' + vn + ' type ' + vt + ' duration ' + dur_str + '. Traits: ' + traits + '. Seasons: ' + seasons + '. Source: ' + sr + '.'
        chunks.append({
            'chunk_id': f'agro_rice_v_c{idx}',
            'text': t,
            'metadata': {'source': filepath.name, 'page': 1, 'domain': AGRO_DOMAIN,
                        'type': 'rice_variety', 'char_count': len(t), 'chunk_index': idx}
        })
        idx += 1
    for rec in data.get('nutrient_management', []):
        details = '; '.join(str(k) + ': ' + str(v) for k, v in rec.items() if k != 'source')
        sr = rec.get('source', 'TNAU')
        t = 'Rice nutrient management: ' + details + '. Source: ' + sr + '.'
        chunks.append({
            'chunk_id': f'agro_rice_n_c{idx}',
            'text': t,
            'metadata': {'source': filepath.name, 'page': 1, 'domain': AGRO_DOMAIN,
                        'type': 'rice_nutrient', 'char_count': len(t), 'chunk_index': idx}
        })
        idx += 1
    return chunks

def save_chunks(chunks, stem):
    p = EVAL_DIR / ('agro_' + stem + '_chunks.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return p

def update_index(doc_id, stem, chunks, cfile):
    ip = EVAL_DIR / 'document_index.json'
    if ip.exists():
        with open(ip, 'r', encoding='utf-8') as f:
            idx = json.load(f)
    else:
        idx = {}
    if doc_id not in idx:
        idx[doc_id] = {
            'doc_id': doc_id,
            'filename': stem + '.json',
            'domain': AGRO_DOMAIN,
            'filepath': str(AGRO_DIR / (stem + '.json')),
            'ingested_at': datetime.now().isoformat(),
            'total_characters': sum(len(c['text']) for c in chunks),
            'total_chunks': len(chunks),
            'chunks_file': cfile.name
        }
        with open(ip, 'w', encoding='utf-8') as f:
            json.dump(idx, f, indent=4, ensure_ascii=False)
        print('  Indexed: ' + doc_id + ' (' + str(len(chunks)) + ' chunks)')
    else:
        print('  Already indexed: ' + doc_id)

def ingested(stem):
    ip = EVAL_DIR / 'document_index.json'
    if not ip.exists():
        return False
    with open(ip, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    return ('agro_' + stem) in idx

def main():
    print('=' * 60)
    print(' Agro Dataset Ingestion - Preprocessing & Indexing')
    print('=' * 60)
    if not AGRO_DIR.exists():
        print('ERROR: ' + str(AGRO_DIR) + ' not found')
        return
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = RAGPipeline()
    summary = []

    for fname in RAG_READY_FILES:
        fp = AGRO_DIR / fname
        if not fp.exists():
            continue
        stem = fp.stem
        if ingested(stem):
            summary.append({'file': fname, 'chunks': '-', 'status': 'Skip(already indexed)'})
            continue
        print('Processing: ' + fname)
        try:
            c = load_rag_ready(fp)
            sp = save_chunks(c, stem)
            update_index('agro_' + stem, stem, c, sp)
            pipeline.ingest_new_documents(c)
            summary.append({'file': fname, 'chunks': len(c), 'status': 'OK'})
            print('  OK - ' + str(len(c)) + ' chunks')
        except Exception as e:
            summary.append({'file': fname, 'chunks': 0, 'status': 'ERR:' + str(e)})
            print('  ERR: ' + str(e))

    for fname, maxr in LARGE_RAG_READY_FILES.items():
        fp = AGRO_DIR / fname
        if not fp.exists():
            continue
        stem = fp.stem
        if ingested(stem):
            summary.append({'file': fname, 'chunks': '-', 'status': 'Skip(already indexed)'})
            continue
        print('Processing (sample ' + str(maxr) + '): ' + fname)
        try:
            c = load_rag_ready(fp, maxr)
            sp = save_chunks(c, stem)
            update_index('agro_' + stem, stem, c, sp)
            pipeline.ingest_new_documents(c)
            summary.append({'file': fname, 'chunks': len(c), 'status': 'OK(sampled ' + str(maxr) + ')'})
            print('  OK - ' + str(len(c)) + ' chunks')
        except Exception as e:
            summary.append({'file': fname, 'chunks': 0, 'status': 'ERR:' + str(e)})
            print('  ERR: ' + str(e))

    rice = AGRO_DIR / 'rice_dataset.json'
    if rice.exists() and not ingested(rice.stem):
        print('Processing: rice_dataset.json')
        try:
            c = preprocess_rice(rice)
            sp = save_chunks(c, rice.stem)
            update_index('agro_' + rice.stem, rice.stem, c, sp)
            pipeline.ingest_new_documents(c)
            summary.append({'file': 'rice_dataset.json', 'chunks': len(c), 'status': 'OK'})
            print('  OK - ' + str(len(c)) + ' chunks')
        except Exception as e:
            summary.append({'file': 'rice_dataset.json', 'chunks': 0, 'status': 'ERR:' + str(e)})
            print('  ERR: ' + str(e))

    print('-' * 60)
    for s in summary:
        print(s['file'].ljust(42) + str(s['chunks']).rjust(6) + '  ' + s['status'])
    total = sum(int(s['chunks']) for s in summary if str(s['chunks']).isdigit())
    ok = sum(1 for s in summary if 'OK' in str(s['status']))
    print('Done: ' + str(ok) + '/' + str(len(summary)) + ' files, ' + str(total) + ' total new chunks ingested.')

if __name__ == '__main__':
    main()
