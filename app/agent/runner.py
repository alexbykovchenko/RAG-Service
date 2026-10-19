"""CLI-запуск агента без HTTP-сервиса.

Пример:
    python -m app.agent.runner "What is Ridge regression?"

Строит тот же граф, что и FastAPI-сервис, прогоняет один запрос и печатает
финальный ответ, число итераций и затраченное время. Удобно проверять новый
промпт или инструмент, не поднимая uvicorn/Gradio.
"""
import sys
import time

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent_graph


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app.agent.runner "your question"')
        raise SystemExit(2)

    question = " ".join(sys.argv[1:])
    graph = build_agent_graph(checkpointer=MemorySaver())

    t0 = time.perf_counter()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "iteration_count": 0},
        config={"configurable": {"thread_id": "cli"}},
    )
    elapsed = time.perf_counter() - t0

    answer = result["messages"][-1].content
    iterations = result.get("iteration_count", 0)

    print(answer)
    print(f"\n--- iterations: {iterations} | time: {elapsed:.1f}s ---")


if __name__ == "__main__":
    main()
