import uuid
from app.graph.graph_builder import app_graph


def test_graph_no_document():
    """
    Test the full graph with a simple query and no document.
    Should route through web search and return a final response.
    """

    initial_state = {
        "user_query": "What are my basic rights if I am arrested?",
        "uploaded_doc_path": None,
        "jurisdiction": "Pakistan",
        "user_role": "citizen",
        "matter_type": "criminal",
        "doc_date": None,
        "doc_validated": False,
        "sub_questions": [],
        "source_mode": "",
        "retrieved_chunks": [],
        "retrieval_scores": [],
        "has_confident_retrieval": False,
        "web_search_results": [],
        "staleness_warning": False,
        "amendment_summary": None,
        "generated_answer": "",
        "citations": [],
        "grounding_result": "",
        "confidence": "",
        "unsupported_claims": [],
        "is_high_stakes": False,
        "escalation_message": None,
        "lawyer_questions": None,
        "fallback_triggered": False,
        "parametric_knowledge_used": False,
        "session_id": str(uuid.uuid4()),
        "stm": [],
        "ltm_profile": {},
        "final_response": "",
    }

    result = app_graph.invoke(initial_state)

    # Basic assertions
    assert result is not None
    assert result["final_response"] != ""
    assert result["source_mode"] in ("RAG", "WEB", "BOTH")
    assert result["confidence"] in ("HIGH", "LOW")

    print("\n--- GRAPH TEST RESULT ---")
    print(f"Source mode   : {result['source_mode']}")
    print(f"Jurisdiction  : {result['jurisdiction']}")
    print(f"Matter type   : {result['matter_type']}")
    print(f"Confidence    : {result['confidence']}")
    print(f"High stakes   : {result['is_high_stakes']}")
    print(f"Staleness     : {result['staleness_warning']}")
    print(f"Fallback      : {result['fallback_triggered']}")
    print(f"\nFinal Response:\n{result['final_response']}")


if __name__ == "__main__":
    test_graph_no_document()