from langchain_core.tools import tool
from app.rag.chain import build_rag_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from app.config import settings

import ast


def _print_last_expression(code: str) -> str:
    """Wrap the final bare expression in print(...) so its value is returned.

    PythonREPLTool captures stdout only; ``x = 1 + 1`` (or a trailing ``x``)
    would otherwise return an empty string and leave the agent guessing.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last_expr = tree.body[-1].value
    if (
        isinstance(last_expr, ast.Call)
        and isinstance(last_expr.func, ast.Name)
        and last_expr.func.id == "print"
    ):
        return code
    tree.body[-1] = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[last_expr],
            keywords=[],
        )
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

_python_repl = PythonREPLTool()
_ddg_search = DuckDuckGoSearchRun()


_chain = None
_retriever = None


def _ensure_rag_ready():
    global _chain, _retriever
    if _chain is None:
        _chain, _retriever = build_rag_chain()


@tool
def documentation_search(query: str) -> str:
    """Search the scikit-learn documentation corpus for a given query.

    Use this tool when the user asks about scikit-learn classes, methods,
    parameters, or general ML concepts (Ridge, Lasso, decision trees, metrics).

    Args:
        query: natural-language question or keyword search.

    Returns:
        Answer text followed by a list of source URLs.
    """
    _ensure_rag_ready()
    docs = _retriever.invoke(query)
    answer = _chain.invoke(query)
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    sources_block = "\n".join(f"- {url}" for url in sources)
    return f"{answer}\n\nSources:\n{sources_block}"

# SECURITY NOTE: PythonREPLTool executes arbitrary code in the same process
# as our app. It is NEVER exposed directly to end users — only the agent
# decides what to send into it. For production with untrusted users, wrap
# in a sandbox (gvisor, firecracker, e2b.dev).

@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output.

    Use this tool for arithmetic, computing formulas, or transforming data.
    The value of the last expression is printed automatically, so an explicit
    `print()` is optional.

    Args:
        code: Python source code to execute.

    Returns:
        Stdout of the executed code, or error message.
    """
    return _python_repl.run(_print_last_expression(code))


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo for recent or general-knowledge info.

    Use this tool when the question requires fresh data (release notes,
    latest versions, news) that is NOT in the documentation corpus.

    Args:
        query: free-text search query.

    Returns:
        Top-3 search results as title + snippet text.
    """
    if not settings.enable_web_search:
        return "Web search is disabled in this environment."
    return _ddg_search.run(query)


TOOLS = [documentation_search, python_repl, web_search]