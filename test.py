from app.rag.chain import build_rag_chain
chain, retriever = build_rag_chain()

answer = chain.invoke("What is the difference between Lasso and Ridge?")
print(answer)

for q in [
    "How do I tune the alpha parameter in Ridge?",
    "Tell me what is the regularization strength parameter",
    "How does PyTorch handle gradients?",  # не из корпуса
]:
    print(f"\nQ: {q}")
    print(f"A: {chain.invoke(q)[:200]}...")