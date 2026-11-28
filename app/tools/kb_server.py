"""MCP server: search_contracts, search_playbook (RAG over ChromaDB) (Step 6)."""
from mcp.server.fastmcp import FastMCP

from ingest.contracts import get_collection

mcp = FastMCP("kb")


@mcp.tool()
def search_contracts(customer_id: str, query: str, top_k: int = 3) -> dict:
    """Search ONE customer's contract for clauses matching the query (SLA, penalty, grace, damage).
    Always filtered by customer. Returns clauses with citations."""
    res = get_collection("contracts").query(
        query_texts=[query], n_results=top_k, where={"customer_id": customer_id})
    if not res["documents"] or not res["documents"][0]:
        return {"error": "NO_CONTRACT_FOUND", "customer_id": customer_id}
    clauses = [
        {"citation": f"{m['contract_id']} \u00a7{m['clause_no']}", "text": d}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]
    return {"customer_id": customer_id, "clauses": clauses}


@mcp.tool()
def search_playbook(query: str, top_k: int = 2) -> dict:
    """Search the internal exception-handling playbook for rules relevant to the situation."""
    res = get_collection("playbook").query(query_texts=[query], n_results=top_k)
    return {"rules": res["documents"][0] if res["documents"] else []}


if __name__ == "__main__":
    mcp.run()