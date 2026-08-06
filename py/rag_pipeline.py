import os
import json
import re
import numpy as np
from pathlib import Path
import config
from language_detector import get_response_language_instruction

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingManager:
    def __init__(self, api_key=None, use_local=True):
        self.api_key = api_key or config.get_gemini_api_key()
        self.use_local = use_local
        self.local_model = None

        if self.use_local:
            if SentenceTransformer:
                model_name = config.MULTILINGUAL_EMBEDDING_MODEL
                print(f"Loading multilingual embedding model: {model_name}...")
                try:
                    # Force HuggingFace to load from cached weights offline to prevent network hang
                    import os
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    self.local_model = SentenceTransformer(model_name)
                    print(f"Loaded local model {model_name} successfully.")
                except Exception as e:
                    print(f"Error loading SentenceTransformer({model_name}): {e}. Falling back to default embedding method.")
                    self.use_local = False
            else:
                print("Warning: sentence-transformers not installed. Falling back to Gemini/Dummy Embeddings.")
                self.use_local = False

        if not self.use_local and self.api_key:
            if genai:
                genai.configure(api_key=self.api_key)
            else:
                print("Warning: google-generativeai SDK not installed.")

    def get_embedding(self, text, is_query=False):
        """Generate embedding vector for text using BAAI/bge-m3 if available."""
        if self.use_local and self.local_model:
            try:
                emb = self.local_model.encode(text)
                return emb.tolist() if hasattr(emb, "tolist") else list(emb)
            except Exception as e:
                print(f"Error generating local embedding: {e}")
            
        if self.api_key and genai:
            try:
                task_type = "retrieval_query" if is_query else "retrieval_document"
                result = genai.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    content=text,
                    task_type=task_type
                )
                return result['embedding']
            except Exception as e:
                print(f"Error generating Gemini embedding: {e}")
                
        return self._generate_dummy_embedding(text, dimension=config.MULTILINGUAL_EMBEDDING_DIM)

    def _generate_dummy_embedding(self, text, dimension=1024):
        """Deterministic dummy embedding generator for fallback offline mode."""
        np.random.seed(hash(text) % (2**32 - 1))
        vec = np.random.randn(dimension)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def get_embeddings_batch(self, texts, is_query=False):
        """Generate embeddings for a list of texts in batch mode to maximize speed on CPU."""
        if self.use_local and self.local_model:
            try:
                # encode takes a list of texts and returns a numpy matrix of embeddings
                embs = self.local_model.encode(texts, batch_size=32, show_progress_bar=False)
                return embs.tolist() if hasattr(embs, "tolist") else [list(e) for e in embs]
            except Exception as e:
                print(f"Error generating local batch embeddings: {e}")
                
        # Fallback to individual embeddings if any issues occur
        return [self.get_embedding(t, is_query) for t in texts]

class SimpleVectorStore:
    def __init__(self, db_dir=None):
        self.db_dir = Path(db_dir or config.VECTOR_DB_DIR)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "vector_store_index.json"
        self.embeddings_file = self.db_dir / "embeddings.npy"
        
        self.chunks_data = [] # List of dict: {chunk_id, text, metadata}
        self.embeddings = [] # numpy matrix
        self.load()

    def add_chunks(self, chunks, embedding_manager):
        """Generate embeddings and add chunks to the vector store."""
        print(f"Embedding {len(chunks)} chunks in batch mode...")
        texts = [chunk["text"] for chunk in chunks]
        new_embeddings = embedding_manager.get_embeddings_batch(texts, is_query=False)
            
        # Append to existing
        self.chunks_data.extend(chunks)
        new_arr = np.array(new_embeddings, dtype=np.float32)
        if len(self.embeddings) == 0:
            self.embeddings = new_arr
        else:
            self.embeddings = np.vstack([self.embeddings, new_arr])
            
        self.save()
        
        # Pre-normalize in memory
        if len(self.embeddings) > 0:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            self.embeddings = self.embeddings / norms
            
        print("Vector store updated and saved.")

    def save(self):
        """Save vector store data to files."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks_data, f, indent=4, ensure_ascii=False)
        if len(self.embeddings) > 0:
            np.save(self.embeddings_file, self.embeddings)

    def load(self):
        """Load vector store data from files."""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.chunks_data = json.load(f)
        if self.embeddings_file.exists():
            embeddings = np.load(self.embeddings_file)
            # Pre-normalize embeddings to avoid expensive np.linalg.norm allocations during query
            if len(embeddings) > 0:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-9, norms)
                self.embeddings = embeddings / norms
            else:
                self.embeddings = embeddings

    def clear(self):
        """Clear all stored data."""
        self.chunks_data = []
        self.embeddings = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()

    def keyword_overlap_score(self, query, doc_text):
        """Compute simple keyword overlap score for hybrid retrieval."""
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', doc_text.lower()))
        if not query_words:
            return 0.0
        intersection = query_words.intersection(doc_words)
        return len(intersection) / len(query_words)

    def query(self, query_text, embedding_manager, top_k=3, hybrid_weight=0.7):
        """
        Query the vector store with multilingual capability using cosine similarity.
        """
        if not self.chunks_data or len(self.embeddings) == 0:
            return []

        # Get query embedding
        query_emb = np.array(embedding_manager.get_embedding(query_text, is_query=True), dtype=np.float32)
        
        # Normalize query vector to unit length
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm
            
        # Cosine similarity is now a simple, zero-allocation dot product (A . B) since arrays are pre-normalized
        vector_scores = np.dot(self.embeddings, query_emb)

        # Normalize vector scores to 0-1 range (approximately)
        vector_scores = (vector_scores + 1) / 2

        results = []
        for idx, chunk in enumerate(self.chunks_data):
            vec_score = float(vector_scores[idx])
            kw_score = self.keyword_overlap_score(query_text, chunk["text"])
            
            # Combine scores
            combined_score = (hybrid_weight * vec_score) + ((1 - hybrid_weight) * kw_score)
            
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "vector_score": vec_score,
                "keyword_score": kw_score,
                "score": combined_score
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

class RAGPipeline:
    def __init__(self, api_key=None, use_local_embeddings=True):
        self.emb_manager = EmbeddingManager(api_key=api_key, use_local=use_local_embeddings)
        self.vector_store = SimpleVectorStore()

    def ingest_new_documents(self, doc_chunks):
        """Index newly ingested chunks into the vector store."""
        if doc_chunks:
            self.vector_store.add_chunks(doc_chunks, self.emb_manager)

    def retrieve(self, query_text, top_k=3, hybrid_weight=0.7):
        """Retrieve relevant context chunks."""
        return self.vector_store.query(query_text, self.emb_manager, top_k, hybrid_weight)

    def generate_enhanced_response(self, query_text, context_chunks, api_key=None, target_language="en"):
        """Generate response in target language with injected context using Gemini."""
        key = api_key or self.emb_manager.api_key
        if not key or not genai:
            return f"Error: Gemini API key is missing or SDK is not loaded. Cannot generate response in {target_language}.", 0.0

        genai.configure(api_key=key)
        
        # Format context
        context_str = "\n\n".join([
            f"[Source: {c['metadata']['source']}, Page: {c['metadata']['page']}]\n{c['text']}" 
            for c in context_chunks
        ])
        
        lang_instruction = get_response_language_instruction(target_language)
        
        prompt = f"""You are an expert advisor specializing in regional, cultural, and traditional knowledge. 
You are provided with verified reference context from domain manuals below. 

REFERENCE CONTEXT:
{context_str}

USER QUESTION:
{query_text}

INSTRUCTIONS:
1. Generate an accurate, comprehensive, and helpful response to the user's question based strictly on the provided REFERENCE CONTEXT. 
2. If the reference context does not contain enough information to answer, state that clearly and do not hallucinate details.
3. Represent the traditional terminology and specifications precisely as documented.
4. {lang_instruction}
"""
        try:
            model = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
            response = model.generate_content(prompt)
            confidence = float(np.mean([c["score"] for c in context_chunks])) if context_chunks else 0.5
            return response.text, confidence
        except Exception as e:
            return f"Error invoking Gemini API: {e}", 0.0

    def generate_baseline_response(self, query_text, api_key=None, target_language="en"):
        """Generate baseline response in target language without RAG context using Gemini."""
        key = api_key or self.emb_manager.api_key
        if not key or not genai:
            return "Error: Gemini API key is missing or SDK is not loaded. Cannot generate response."

        genai.configure(api_key=key)
        
        lang_instruction = get_response_language_instruction(target_language)
        
        prompt = f"""Answer the following question to the best of your ability. Keep your answer brief and factual.
{lang_instruction}

Question: {query_text}
Answer:"""
        try:
            model = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error invoking Gemini API for baseline: {e}"
