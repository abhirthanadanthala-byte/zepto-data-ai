from fastapi import FastAPI

from .graph import graph
from .schemas import AskRequest, AssistantResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant API is running"
    }


@app.post("/ask", response_model=AssistantResponse)
def ask(request: AskRequest):
    """
    Process a user query through the LangGraph workflow.
    """

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return AssistantResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )