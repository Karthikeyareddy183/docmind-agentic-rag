"""MCP server: use your DocMind documents from Claude Desktop, Cursor, or any MCP client.

It is a thin client of the DocMind REST API, so authentication, per-user isolation, and
retrieval quality are exactly the same as in the web app.

Configuration (environment variables):
    DOCMIND_API_URL     default http://localhost:8000
    DOCMIND_EMAIL       DocMind account email
    DOCMIND_PASSWORD    DocMind account password

Run:  python -m docmind.mcp_server            (stdio transport)

Claude Desktop / Claude Code config::

    {"mcpServers": {"docmind": {
        "command": "D:/docmind-env/venv/Scripts/python.exe",
        "args": ["-m", "docmind.mcp_server"],
        "env": {"DOCMIND_EMAIL": "...", "DOCMIND_PASSWORD": "..."}}}}
"""

import os
import time
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

API_URL = os.environ.get("DOCMIND_API_URL", "http://localhost:8000").rstrip("/") + "/api/v1"
TIMEOUT = httpx.Timeout(180.0, connect=10.0)

server = MCPServer(
    name="docmind",
    title="DocMind",
    instructions=(
        "Search and question the user's uploaded documents (PDFs, scans, images). "
        "Use list_documents to see what is available, search_documents for raw passages, "
        "and ask_documents for a cited answer. Always relay the page citations to the user."
    ),
)


class _Session:
    """Logs in lazily and refreshes the access token shortly before it expires."""

    def __init__(self) -> None:
        self.token: str | None = None
        self.expires_at = 0.0

    async def headers(self, client: httpx.AsyncClient) -> dict[str, str]:
        if self.token is None or time.time() > self.expires_at - 60:
            email, password = os.environ.get("DOCMIND_EMAIL"), os.environ.get("DOCMIND_PASSWORD")
            if not email or not password:
                raise RuntimeError(
                    "Set DOCMIND_EMAIL and DOCMIND_PASSWORD for the DocMind MCP server"
                )
            resp = await client.post(
                f"{API_URL}/auth/login", data={"username": email, "password": password}
            )
            resp.raise_for_status()
            body = resp.json()
            self.token = body["access_token"]
            self.expires_at = time.time() + int(body.get("expires_in", 3600))
        return {"Authorization": f"Bearer {self.token}"}


_session = _Session()


async def _request(method: str, path: str, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.request(
            method, f"{API_URL}{path}", headers=await _session.headers(client), **kwargs
        )
        if resp.status_code >= 400:
            try:
                message = resp.json()["error"]["message"]
            except Exception:
                message = resp.text[:300]
            raise RuntimeError(f"DocMind API error {resp.status_code}: {message}")
        return resp.json()


@server.tool(description="List the user's documents with their processing status.")
async def list_documents() -> list[dict[str, Any]]:
    data = await _request("GET", "/documents", params={"limit": 100})
    return [
        {
            "id": d["id"],
            "title": d.get("title") or d["filename"],
            "filename": d["filename"],
            "status": d["status"],
            "pages": d.get("pages"),
            "summary": d.get("summary"),
        }
        for d in data["items"]
    ]


@server.tool(
    description="Hybrid semantic + keyword search over the user's documents. Returns the most "
    "relevant passages with document, page, and relevance score (no LLM answer)."
)
async def search_documents(
    query: str, document_ids: list[str] | None = None, limit: int = 6
) -> list[dict[str, Any]]:
    data = await _request(
        "POST", "/search", json={"query": query, "document_ids": document_ids or [], "limit": limit}
    )
    return [
        {
            "document": h["document_title"],
            "page": h["page"],
            "section": h["section"],
            "type": h["chunk_type"],
            "score": h["score"],
            "text": h["content"],
        }
        for h in data["hits"]
    ]


@server.tool(
    description="Answer a question from the user's documents. Returns an answer with [S#] "
    "citations (document + page), a computed confidence score, and whether it was fact-checked."
)
async def ask_documents(question: str, document_ids: list[str] | None = None) -> dict[str, Any]:
    data = await _request(
        "POST", "/ask", json={"question": question, "document_ids": document_ids or []}
    )
    return {
        "answer": data["answer"],
        "confidence": data["confidence"],
        "verified": data["verified"],
        "citations": [
            {
                "id": c["id"],
                "document": c["document_title"],
                "page": c["page"],
                "snippet": c["snippet"][:240],
            }
            for c in data["citations"]
        ],
    }


if __name__ == "__main__":
    server.run("stdio")
