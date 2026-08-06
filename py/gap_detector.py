import json
import re
import numpy as np
import config

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class KnowledgeGapDetector:
    def __init__(self, embedding_manager):
        self.emb_manager = embedding_manager

    def compute_semantic_similarity(self, text_a, text_b):
        """Compute cosine similarity between two texts using the embedding manager."""
        emb_a = np.array(self.emb_manager.get_embedding(text_a, is_query=False))
        emb_b = np.array(self.emb_manager.get_embedding(text_b, is_query=False))
        
        dot_product = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        similarity = dot_product / (norm_a * norm_b)
        # Map cosine similarity [-1, 1] to [0, 1] range
        similarity_mapped = (similarity + 1) / 2
        return float(np.clip(similarity_mapped, 0.0, 1.0))

    def analyze_concepts_fallback(self, expected_answer, llm_answer):
        """
        Local fallback to analyze concept coverage when Gemini API is unavailable.
        Uses simple keyword matching, but falls back to cross-lingual semantic similarity
        if the response is in Hindi/Kannada to prevent artificial 100% gap scores.
        """
        # Split expected answer into sentences/clauses as rough concepts
        sentences = [s.strip() for s in re.split(r'[;.]', expected_answer) if len(s.strip()) > 15]
        if not sentences:
            sentences = [expected_answer]
            
        # Compute general semantic similarity between expected answer and LLM answer
        sim = self.compute_semantic_similarity(expected_answer, llm_answer)
        
        concepts = []
        for sent in sentences[:6]: # limit to top 6 sentences/concepts
            # Extract key nouns/terms for a simple match
            words = [w.lower() for w in re.findall(r'\b\w{4,}\b', sent)]
            
            # Simple keyword matching (only works if same language)
            has_keyword_match = False
            match_count = 0
            if words:
                match_count = sum(1 for w in words if w in llm_answer.lower())
                has_keyword_match = (match_count / len(words)) > 0.3
            
            # For multilingual outputs, if semantic similarity is high (e.g. > 0.74),
            # we infer that the concepts are semantically present.
            if sim > 0.74 or has_keyword_match:
                status = "Present"
                evidence = f"Concept verified present via cross-lingual semantic similarity ({sim:.3f})."
            else:
                status = "Absent"
                evidence = f"Low semantic similarity ({sim:.3f}) and keyword matches ({match_count}/{len(words) if words else 0})."
                
            concepts.append({
                "concept": sent,
                "status": status,
                "evidence": evidence
            })
            
        return concepts

    def analyze_concepts_with_gemini(self, expected_answer, llm_answer, api_key=None):
        """Use Gemini to extract key concepts and check if they are present in LLM answer."""
        key = api_key or self.emb_manager.api_key
        if not key or not genai:
            return self.analyze_concepts_fallback(expected_answer, llm_answer)

        genai.configure(api_key=key)
        
        prompt = f"""You are a high-fidelity evaluation agent for Responsible AI.
Compare the LLM Answer against the Expected Answer to identify missing knowledge concepts.

EXPECTED ANSWER (in English):
{expected_answer}

LLM ANSWER (which may be in English, Hindi, or Kannada):
{llm_answer}

Identify 3 to 6 distinct, essential core concepts (facts, quantities, ingredients, or steps) present in the EXPECTED ANSWER.
For each concept, evaluate whether it is accurately and sufficiently present in the LLM ANSWER (mark as "Present" or "Absent").
Since the LLM ANSWER may be in Hindi or Kannada, evaluate semantic presence across languages. Be strict. If a quantity is wrong or a key step is omitted, mark it as "Absent".

OUTPUT FORMAT:
Provide the output in strict JSON format as a list of objects. Do not wrap the JSON in ```json markdown formatting.
Each object must have exactly these keys:
"concept": string (description of the specific concept),
"status": string ("Present" or "Absent"),
"evidence": string (brief quote or reasoning for the decision)

Example JSON Output:
[
  {{
    "concept": "Mix cow dung and ghee for 3 days",
    "status": "Present",
    "evidence": "LLM answer mentions fermenting the dung and ghee combination for three days in Hindi/Kannada."
  }}
]
"""
        try:
            model = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"} if hasattr(genai, "GenerationConfig") else None
            )
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"^```json\s*", "", cleaned_text)
                cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            return json.loads(cleaned_text)
        except Exception as e:
            print(f"Error calling Gemini in concept analysis: {e}. Using local fallback.")
            return self.analyze_concepts_fallback(expected_answer, llm_answer)

    def evaluate_gap(self, question, expected_answer, llm_answer, api_key=None):
        """
        Evaluate semantic gap and compute the custom Knowledge Gap Index (KGI).
        Returns a dictionary containing all scores, concepts analysis, and index.
        """
        # 1. Semantic Similarity
        semantic_similarity = self.compute_semantic_similarity(expected_answer, llm_answer)
        semantic_gap = 1.0 - semantic_similarity

        # 2. Concept Analysis
        concepts = self.analyze_concepts_with_gemini(expected_answer, llm_answer, api_key)
        
        # Calculate coverage and missing concept scores
        total_concepts = len(concepts) if concepts else 1
        present_concepts = sum(1 for c in concepts if c["status"] == "Present")
        absent_concepts = total_concepts - present_concepts
        
        coverage_score = present_concepts / total_concepts
        coverage_gap = 1.0 - coverage_score
        missing_concept_score = absent_concepts / total_concepts

        # 3. Compute custom Knowledge Gap Index (KGI)
        # KGI = 0.4 * Semantic Gap + 0.3 * Coverage Gap + 0.3 * Missing Concept Gap
        weights = config.KGI_WEIGHTS
        kgi = (
            weights["semantic"] * semantic_gap +
            weights["coverage"] * coverage_gap +
            weights["concept"] * missing_concept_score
        )
        
        # Clip KGI to [0.0, 1.0]
        kgi = float(np.clip(kgi, 0.0, 1.0))

        return {
            "question": question,
            "expected_answer": expected_answer,
            "llm_answer": llm_answer,
            "semantic_similarity": semantic_similarity,
            "semantic_gap": semantic_gap,
            "coverage_score": coverage_score,
            "coverage_gap": coverage_gap,
            "missing_concept_score": missing_concept_score,
            "kgi": kgi,
            "concepts": concepts
        }
