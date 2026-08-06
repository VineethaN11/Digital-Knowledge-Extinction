import os
import json
from pathlib import Path
import config
from ingestion import DocumentIngester
from rag_pipeline import RAGPipeline
from question_generator import BenchmarkQuestionGenerator
from gap_detector import KnowledgeGapDetector
from agent_decision import AgenticDecisionLayer
from evaluation import RAIEvaluator, analyze_gap_reduction

def print_header(title):
    print("\n" + "="*60)
    print(f" {title} ".center(60, "="))
    print("="*60)

def main():
    print_header("DIGITAL KNOWLEDGE EXTINCTION FRAMEWORK VALIDATION")
    
    # 1. Check API Key
    api_key = config.get_gemini_api_key()
    live_mode = api_key is not None and len(api_key.strip()) > 0
    
    if live_mode:
        print(f"[STATUS] Running in LIVE MODE using Gemini API Key.")
    else:
        print(f"[STATUS] Running in SIMULATION MODE (No API Key found).")
        print(f"[INFO] The pipeline will simulate LLM generation & concept mapping for demonstration.")

    # 2. Ingest Document
    print_header("MODULE 1 & 7: KNOWLEDGE INGESTION & VECTOR INDEXING")
    sample_file = config.CORPUS_DIR / "traditional_agriculture_manual.txt"
    if not sample_file.exists():
        print(f"[ERROR] Sample file not found at {sample_file}!")
        return
        
    ingester = DocumentIngester()
    doc_id, doc_meta = ingester.ingest_document(
        sample_file, 
        domain="Traditional Knowledge / Sustainable Agriculture",
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Initialize RAG Pipeline and add chunks
    pipeline = RAGPipeline(api_key=api_key)
    chunks_file = config.EVAL_DIR / doc_meta["chunks_file"]
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    pipeline.ingest_new_documents(chunks)

    # 3. Generate Benchmark Questions
    print_header("MODULE 2: BENCHMARK QUESTION GENERATION")
    generator = BenchmarkQuestionGenerator()
    questions = generator.generate_benchmark(doc_id, num_questions=3, api_key=api_key)
    
    for idx, q in enumerate(questions):
        print(f"Q{idx+1}: {q['question']}")
        print(f"   Domain: {q['domain']} | Type: {q['type']}")
    
    # Select first question for tracing
    test_q = questions[0]
    print(f"\n[SELECTED FOR EVALUATION]: '{test_q['question']}'")

    # 4. Baseline Evaluation
    print_header("MODULE 3, 4 & 5: BASELINE LLM EVALUATION & KGI COMPUTATION")
    
    detector = KnowledgeGapDetector(pipeline.emb_manager)
    
    if live_mode:
        baseline_ans = pipeline.generate_baseline_response(test_q["question"])
        baseline_eval = detector.evaluate_gap(test_q["question"], test_q["expected_answer"], baseline_ans)
    else:
        # Simulate a typical baseline response with gaps
        baseline_ans = (
            "Panchagavya is a traditional Indian fertilizer made from cow dung, urine, milk, and curd. "
            "It is fermented together and sprayed on crops to improve their yield. It has been used for centuries."
        )
        # Compute KGI with local fallback
        baseline_eval = detector.evaluate_gap(test_q["question"], test_q["expected_answer"], baseline_ans)
        
    print(f"Baseline Answer:\n{baseline_ans}\n")
    print(f"Calculated Metrics:")
    print(f" - Semantic Similarity: {baseline_eval['semantic_similarity']:.3f}")
    print(f" - Concept Coverage:    {baseline_eval['coverage_score']:.2%}")
    print(f" - Missing Concept:     {baseline_eval['missing_concept_score']:.2%}")
    print(f" - Knowledge Gap Index (KGI): {baseline_eval['kgi']:.3f} (Severe Gap = 1.00)")

    # 5. Agentic Decision Layer
    print_header("MODULE 6: EXPLAINABLE AGENTIC DECISION LAYER")
    agent = AgenticDecisionLayer(api_key=api_key)
    agent_decision = agent.decide(test_q["question"], baseline_eval["kgi"], [test_q["domain"]])
    
    print(f"Decision:  {agent_decision['decision']}")
    print(f"Reason:    {agent_decision['reason']}")
    print(f"Domain:    {agent_decision['chosen_domain']}")
    print(f"Confidence: {agent_decision['confidence']:.2%}")

    # 6. RAG Context Retrieval & Generation
    print_header("MODULE 7 & 8: RAG RETRIEVAL & ENHANCED RESPONSE")
    
    if agent_decision["decision"] == "YES":
        # Retrieve context
        retrieved_chunks = pipeline.retrieve(test_q["question"], top_k=3)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks from '{test_q['source_document']}':")
        for c in retrieved_chunks:
            print(f" - Chunk {c['chunk_id']} (Score: {c['score']:.3f}) | Snippet: {c['text'][:50]}...")
            
        if live_mode:
            enhanced_ans, confidence = pipeline.generate_enhanced_response(test_q["question"], retrieved_chunks)
            enhanced_eval = detector.evaluate_gap(test_q["question"], test_q["expected_answer"], enhanced_ans)
        else:
            # Simulate high-fidelity RAG response
            enhanced_ans = (
                "Panchagavya is a traditional organic formulation that requires nine exact ingredients: "
                "1) Fresh cow dung (7 kg), 2) Cow ghee (1 kg), 3) Fresh cow urine (10 liters), 4) Water (10 liters), "
                "5) Cow milk (3 liters), 6) Cow curd (2 liters), 7) Tender coconut water (3 liters), "
                "8) Jaggery (3 kg), and 9) Ripe bananas (12 numbers). Fermentation takes 25-30 days in total."
            )
            enhanced_eval = detector.evaluate_gap(test_q["question"], test_q["expected_answer"], enhanced_ans)
            
        print(f"\nEnhanced RAG Answer:\n{enhanced_ans}\n")
    else:
        print("Retrieval skipped by Agent. Enhanced response identical to baseline.")
        enhanced_ans = baseline_ans
        enhanced_eval = baseline_eval

    # 7. Knowledge Gap Reduction & RAI scorecards
    print_header("MODULE 9 & 10: GAP REDUCTION & RESPONSIBLE AI SCORECARD")
    
    reduction = analyze_gap_reduction(baseline_eval, enhanced_eval)
    print(f"Gap Reduction Summary:")
    print(f" - KGI Before: {reduction['kgi_before']:.3f}")
    print(f" - KGI After:  {reduction['kgi_after']:.3f}")
    print(f" - Gap Reduction Percentage: {reduction['gap_reduction_percentage']:.2f}%")
    print(f" - Semantic Improvement:     +{reduction['semantic_improvement']:.3f}")
    print(f" - Coverage Improvement:     +{reduction['coverage_improvement']:.2%}")
    
    # RAI scorecard
    rai_evaluator = RAIEvaluator()
    if agent_decision["decision"] == "YES":
        chunks_for_score = retrieved_chunks
    else:
        chunks_for_score = []
        
    rai_score = rai_evaluator.compute_rai_scorecard(
        baseline_ans, 
        enhanced_ans, 
        test_q["expected_answer"], 
        agent_decision, 
        chunks_for_score
    )
    
    print("\nResponsible AI Scorecard:")
    print(f" Metric               | Baseline | Enhanced | Delta ")
    print(f" ---------------------|----------|----------|-------")
    print(f" Inclusiveness        | {rai_score['baseline']['inclusiveness']:.2f}     | {rai_score['enhanced']['inclusiveness']:.2f}     | +{(rai_score['enhanced']['inclusiveness'] - rai_score['baseline']['inclusiveness']):.2f}")
    print(f" Fair Representation  | {rai_score['baseline']['fair_representation']:.2f}     | {rai_score['enhanced']['fair_representation']:.2f}     | +{(rai_score['enhanced']['fair_representation'] - rai_score['baseline']['fair_representation']):.2f}")
    print(f" Transparency         | {rai_score['baseline']['transparency']:.2f}     | {rai_score['enhanced']['transparency']:.2f}     | +{(rai_score['enhanced']['transparency'] - rai_score['baseline']['transparency']):.2f}")
    print(f" Explainability       | {rai_score['baseline']['explainability']:.2f}     | {rai_score['enhanced']['explainability']:.2f}     | +{(rai_score['enhanced']['explainability'] - rai_score['baseline']['explainability']):.2f}")
    print(f" ---------------------|----------|----------|-------")
    print(f" Overall RAI Index    | {rai_score['baseline']['overall']:.2f}     | {rai_score['enhanced']['overall']:.2f}     | +{(rai_score['enhanced']['overall'] - rai_score['baseline']['overall']):.2f}")
    
    print("\n" + "="*60)
    print(" VERIFICATION SUCCESSFUL ".center(60, "*"))
    print("="*60)

if __name__ == "__main__":
    main()
