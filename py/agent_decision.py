import json
from typing import Dict, Any, TypedDict, Literal
import config

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

# Define State Schema
class AgentState(TypedDict):
    question: str
    kgi: float
    threshold: float
    retrieval_needed: bool
    chosen_domain: str
    decision: Literal["YES", "NO"]
    reason: str
    confidence: float
    available_domains: list

class AgenticDecisionLayer:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.get_gemini_api_key()
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)

    # NODE 1: Assess Gap Node
    def assess_gap_node(self, state: AgentState) -> Dict[str, Any]:
        kgi = state["kgi"]
        threshold = state["threshold"]
        retrieval_needed = kgi >= threshold
        
        return {
            "retrieval_needed": retrieval_needed,
            "decision": "YES" if retrieval_needed else "NO"
        }

    # NODE 2: Select Source Node
    def select_source_node(self, state: AgentState) -> Dict[str, Any]:
        question = state["question"]
        available_domains = state.get("available_domains", ["Traditional Knowledge"])
        
        # Determine domain via keyword matching or default
        # For simplicity, search keywords in question
        chosen_domain = available_domains[0] if available_domains else "Traditional Knowledge"
        
        question_lower = question.lower()
        for domain in available_domains:
            # Check if domain name keywords are in question
            domain_words = set(domain.lower().split())
            if any(w in question_lower for w in domain_words if len(w) > 3):
                chosen_domain = domain
                break
                
        # Specific overrides for agricultural terms
        agri_terms = ["panchagavya", "bijamrita", "akkadi saalu", "navara", "seed", "farming", "soil"]
        if any(term in question_lower for term in agri_terms):
            for domain in available_domains:
                if "agri" in domain.lower() or "traditional" in domain.lower():
                    chosen_domain = domain
                    break

        return {
            "chosen_domain": chosen_domain
        }

    # NODE 3: Explain Decision Node
    def explain_decision_node(self, state: AgentState) -> Dict[str, Any]:
        question = state["question"]
        kgi = state["kgi"]
        decision = state["decision"]
        chosen_domain = state["chosen_domain"]
        threshold = state["threshold"]
        
        if decision == "YES":
            confidence = min(0.5 + (kgi * 0.5), 0.98) # scale confidence with KGI
        else:
            confidence = min(0.98, 0.5 + ((1.0 - kgi) * 0.5))

        reason = ""
        # If API is available, ask Gemini to write a natural explanation
        if self.api_key and genai:
            prompt = f"""You are an Explainable AI (XAI) agent. 
Explain why the system decided to {"perform RAG retrieval" if decision == "YES" else "skip retrieval and use the baseline LLM answer"} for this question.

Context:
- User Question: "{question}"
- Computed Knowledge Gap Index (KGI): {kgi:.2f} (Threshold: {threshold:.2f})
- Target Knowledge Domain: "{chosen_domain}"

Explain in 2-3 concise sentences:
1. Identify the knowledge gap or adequacy of the baseline.
2. Justify why retrieval from the {chosen_domain} domain is required (or why it can be skipped).
3. State the impact of this decision.

Response should be clear, professional, and directly address the user.
"""
            try:
                model = genai.GenerativeModel(config.DEFAULT_GEMINI_MODEL)
                response = model.generate_content(prompt)
                reason = response.text.strip()
            except Exception as e:
                print(f"Error generating explanation: {e}")

        # Fallback explanation if API fails or is not available
        if not reason:
            if decision == "YES":
                reason = (
                    f"RAG retrieval was triggered because the baseline response resulted in a High Knowledge Gap Index (KGI = {kgi:.2f}), "
                    f"exceeding our threshold of {threshold:.2f}. The question requires specific, underrepresented details "
                    f"from the '{chosen_domain}' domain which the baseline LLM failed to accurately cover."
                )
            else:
                reason = (
                    f"Retrieval was skipped because the baseline response resulted in a Low Knowledge Gap Index (KGI = {kgi:.2f}), "
                    f"which is below the threshold of {threshold:.2f}. The baseline model has adequate general knowledge "
                    f"to answer the question without external documentation."
                )

        return {
            "reason": reason,
            "confidence": float(confidence)
        }

    def run_workflow_local(self, initial_state: AgentState) -> AgentState:
        """Run nodes sequentially when LangGraph is not installed (fallback)."""
        state = initial_state.copy()
        
        # Execute Node 1
        res1 = self.assess_gap_node(state)
        state.update(res1)
        
        # Execute Node 2
        res2 = self.select_source_node(state)
        state.update(res2)
        
        # Execute Node 3
        res3 = self.explain_decision_node(state)
        state.update(res3)
        
        return state

    def run_workflow_langgraph(self, initial_state: AgentState) -> AgentState:
        """Run workflow using official LangGraph StateGraph."""
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("assess_gap", self.assess_gap_node)
        workflow.add_node("select_source", self.select_source_node)
        workflow.add_node("explain_decision", self.explain_decision_node)
        
        # Set Entry Point
        workflow.set_entry_point("assess_gap")
        
        # Add Directed Edges
        # From assess_gap, go to select_source
        workflow.add_edge("assess_gap", "select_source")
        # From select_source, go to explain_decision
        workflow.add_edge("select_source", "explain_decision")
        # From explain_decision, go to END
        workflow.add_edge("explain_decision", END)
        
        # Compile
        app = workflow.compile()
        
        # Run
        return app.invoke(initial_state)

    def decide(self, question: str, kgi: float, available_domains: list) -> Dict[str, Any]:
        """
        Executes the agent workflow and returns the decision.
        """
        initial_state: AgentState = {
            "question": question,
            "kgi": kgi,
            "threshold": config.KGI_THRESHOLD,
            "retrieval_needed": False,
            "chosen_domain": "",
            "decision": "NO",
            "reason": "",
            "confidence": 0.0,
            "available_domains": available_domains
        }
        
        if HAS_LANGGRAPH:
            try:
                print("Executing agent decision layer using LangGraph...")
                final_state = self.run_workflow_langgraph(initial_state)
            except Exception as e:
                print(f"Error in LangGraph workflow execution: {e}. Falling back to local runner.")
                final_state = self.run_workflow_local(initial_state)
        else:
            print("Executing agent decision layer using Local Sequential Runner...")
            final_state = self.run_workflow_local(initial_state)
            
        return {
            "decision": final_state["decision"],
            "reason": final_state["reason"],
            "chosen_domain": final_state["chosen_domain"],
            "confidence": final_state["confidence"]
        }
