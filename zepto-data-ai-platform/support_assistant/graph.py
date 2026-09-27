from typing import TypedDict

from langgraph.graph import StateGraph, END

from .retrieval import retrieve_documents
from .prompts import build_prompt


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: list
    answer: str
    sources: list[str]
    confidence: float


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: AssistantState):
    """
    Classify the user's query using a simple deterministic
    keyword-based policy/general classifier.
    """

    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


def retrieve_and_answer(state: AssistantState):
    """
    Retrieve the top policy documents and generate the
    deterministic MOCK_LLM answer.
    """

    query = state["query"]

    results = retrieve_documents(query, top_k=3)

    if not results:
        return {
            "retrieved_documents": [],
            "answer": "No relevant Zepto policy information was found.",
            "sources": [],
            "confidence": 0.0,
        }

    top_document = results[0]["document"]
    top_source = results[0]["source"]

    # Build the structured prompt.
    # This prompt is ready for the optional real LLM path.
    _prompt = build_prompt(
        query=query,
        context="\n\n".join(
            result["document"] for result in results
        )
    )

    # Deterministic MOCK_LLM response.
    answer = (
        "Based on the retrieved context: "
        + top_document[:500]
    )

    sources = [
        result["source"]
        for result in results
    ]

    return {
        "retrieved_documents": results,
        "answer": answer,
        "sources": sources,
        "confidence": 1.0,
    }


def direct_answer(state: AssistantState):
    """
    Deterministic response for general questions.
    """

    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0,
    }


def route_after_classification(state: AssistantState):
    """
    Conditional routing after intent classification.
    """

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

workflow = StateGraph(AssistantState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("LANGGRAPH TEST")
    print("=" * 70)

    # Test 1: Policy question
    policy_query = "How long do I have to report damaged items?"

    result = graph.invoke(
        {
            "query": policy_query
        }
    )

    print("\nPOLICY QUESTION:")
    print(policy_query)

    print("\nIntent:")
    print(result["intent"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    print("\nConfidence:")
    print(result["confidence"])

    # Test 2: General question
    general_query = "Tell me a joke."

    result = graph.invoke(
        {
            "query": general_query
        }
    )

    print("\n" + "-" * 70)

    print("\nGENERAL QUESTION:")
    print(general_query)

    print("\nIntent:")
    print(result["intent"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(result["sources"])

    print("\nConfidence:")
    print(result["confidence"])

    print("\n" + "=" * 70)
    print("LANGGRAPH TEST COMPLETED")
    print("=" * 70)