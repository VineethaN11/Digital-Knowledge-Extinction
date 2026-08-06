import re
import config

class RAIEvaluator:
    def __init__(self):
        self.weights = config.RAI_WEIGHTS

    def evaluate_inclusiveness(self, text, expected_answer):
        """
        Measure inclusion of regional/cultural/traditional terminology.
        Extract unique capitalized or specific non-English terms from expected answer 
        and check if they are present in the text.
        """
        # Find potential regional/specific terms (e.g., words like Panchagavya, Akkadi Saalu, Bijamrita, Navara)
        # We can extract terms that are 4+ characters, starting with capital letter or specific terms
        traditional_vocab = {"panchagavya", "bijamrita", "beejamrutha", "akkadi", "saalu", "navara", "njavara", "navarakizhi", "kotta", "moode", "azadirachta"}
        
        # Also extract proper nouns from expected answer
        words_expected = set(re.findall(r'\b[A-Z][a-zA-Z]+\b', expected_answer))
        words_expected_lower = {w.lower() for w in words_expected}
        
        # Merge with pre-defined traditional terms present in expected answer
        target_terms = traditional_vocab.intersection({w.lower() for w in re.findall(r'\b\w+\b', expected_answer)})
        target_terms = target_terms.union(words_expected_lower)
        
        # Exclude standard English proper nouns or short terms if any
        exclude = {"phase", "ingredients", "ingredients:", "preparation", "usage", "akkadi", "traditional"}
        target_terms = {t for t in target_terms if len(t) > 3 and t not in exclude}
        
        if not target_terms:
            return 1.0 # If no specific terminology found, default to full inclusion
            
        found_terms = [t for t in target_terms if t in text.lower()]
        return len(found_terms) / len(target_terms)

    def evaluate_fair_representation(self, text, expected_answer):
        """
        Evaluate if specific numeric values, ratios, and critical facts are represented accurately.
        """
        # Extract numbers (e.g. 7 kg, 10 liters, 3%, 9 rows) from expected answer
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', expected_answer)
        if not numbers:
            return 1.0
            
        # Count how many of these numbers are present in the response
        found_numbers = [num for num in numbers if num in text]
        
        # Let's also check some unit matchings if any
        # This provides a basic proxy for numeric fidelity
        return len(found_numbers) / len(numbers)

    def compute_rai_scorecard(self, baseline_response, enhanced_response, expected_answer, agent_decision, context_chunks, query_lang="en", baseline_lang="en", enhanced_lang="en"):
        """
        Generates the Responsible AI Scorecard comparing baseline and enhanced responses.
        Includes multilingual evaluation metrics.
        """
        # Evaluate Inclusiveness (traditional terms preserved)
        incl_baseline = self.evaluate_inclusiveness(baseline_response, expected_answer)
        incl_enhanced = self.evaluate_inclusiveness(enhanced_response, expected_answer)
        
        # Evaluate Fair Representation (numeric / factual fidelity)
        fair_baseline = self.evaluate_fair_representation(baseline_response, expected_answer)
        fair_enhanced = self.evaluate_fair_representation(enhanced_response, expected_answer)

        # Transparency (are citations/sources declared?)
        trans_baseline = 0.0
        trans_enhanced = 1.0 if len(context_chunks) > 0 else 0.0

        # Explainability (is agent reasoning log present?)
        exp_baseline = 0.0
        exp_enhanced = 1.0 if agent_decision and "reason" in agent_decision else 0.0

        # Multilingual RAI extensions
        # 1. Language Inclusiveness (is response in query language?)
        lang_inc_baseline = 1.0 if baseline_lang == query_lang else 0.5
        lang_inc_enhanced = 1.0 if enhanced_lang == query_lang else 0.5

        # 2. Cross-Lingual Representation (retrieve English content for non-English queries)
        cross_baseline = 0.0
        cross_enhanced = 1.0 if (query_lang in ["hi", "kn"] and len(context_chunks) > 0) else (1.0 if query_lang == "en" else 0.5)

        # 3. Regional Knowledge Accessibility
        access_baseline = 0.3 if query_lang in ["hi", "kn"] else 1.0
        access_enhanced = 1.0 if (query_lang == "en" or len(context_chunks) > 0) else 0.5

        # Weighted Scores
        overall_baseline = (
            incl_baseline * self.weights.get("inclusiveness", 0.15) +
            fair_baseline * self.weights.get("fair_representation", 0.15) +
            trans_baseline * self.weights.get("transparency", 0.15) +
            exp_baseline * self.weights.get("explainability", 0.15) +
            incl_baseline * self.weights.get("knowledge_preservation", 0.15) +
            lang_inc_baseline * self.weights.get("language_inclusiveness", 0.125) +
            cross_baseline * self.weights.get("cross_lingual_representation", 0.05) +
            access_baseline * self.weights.get("regional_knowledge_access", 0.075)
        )

        overall_enhanced = (
            incl_enhanced * self.weights.get("inclusiveness", 0.15) +
            fair_enhanced * self.weights.get("fair_representation", 0.15) +
            trans_enhanced * self.weights.get("transparency", 0.15) +
            exp_enhanced * self.weights.get("explainability", 0.15) +
            incl_enhanced * self.weights.get("knowledge_preservation", 0.15) +
            lang_inc_enhanced * self.weights.get("language_inclusiveness", 0.125) +
            cross_enhanced * self.weights.get("cross_lingual_representation", 0.05) +
            access_enhanced * self.weights.get("regional_knowledge_access", 0.075)
        )

        return {
            "baseline": {
                "inclusiveness": incl_baseline,
                "fair_representation": fair_baseline,
                "transparency": trans_baseline,
                "explainability": exp_baseline,
                "language_inclusiveness": lang_inc_baseline,
                "cross_lingual_representation": cross_baseline,
                "regional_knowledge_access": access_baseline,
                "overall": overall_baseline
            },
            "enhanced": {
                "inclusiveness": incl_enhanced,
                "fair_representation": fair_enhanced,
                "transparency": trans_enhanced,
                "explainability": exp_enhanced,
                "language_inclusiveness": lang_inc_enhanced,
                "cross_lingual_representation": cross_enhanced,
                "regional_knowledge_access": access_enhanced,
                "overall": overall_enhanced
            }
        }

def analyze_gap_reduction(before_kgi_report, after_kgi_report):
    """
    Computes comparative metrics showing how much the knowledge gap was reduced.
    """
    kgi_before = before_kgi_report["kgi"]
    kgi_after = after_kgi_report["kgi"]
    
    # Gap Reduction Percentage
    if kgi_before > 0:
        gap_reduction_pct = ((kgi_before - kgi_after) / kgi_before) * 100
    else:
        gap_reduction_pct = 100.0 if kgi_after == 0 else 0.0
        
    gap_reduction_pct = max(0.0, gap_reduction_pct) # Cap at 0 if gap increased (rare)
    
    # Improvements in individual components
    semantic_improvement = after_kgi_report["semantic_similarity"] - before_kgi_report["semantic_similarity"]
    coverage_improvement = after_kgi_report["coverage_score"] - before_kgi_report["coverage_score"]
    concept_score_reduction = before_kgi_report["missing_concept_score"] - after_kgi_report["missing_concept_score"]

    return {
        "kgi_before": kgi_before,
        "kgi_after": kgi_after,
        "gap_reduction_percentage": gap_reduction_pct,
        "semantic_improvement": semantic_improvement,
        "coverage_improvement": coverage_improvement,
        "concept_score_reduction": concept_score_reduction,
        "accuracy_improvement": (semantic_improvement + coverage_improvement) / 2
    }
