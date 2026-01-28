from typing import Any, Dict, List


class RemoteContext:
    def __init__(self, tools):
        self._tools = tools

    def bm25_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return self._tools.bm25_search(query=query, k=k)

    def vector_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return self._tools.vector_search(query=query, k=k)

    def hybrid_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return self._tools.hybrid_search(query=query, k=k)

    def fetch_chunk(self, chunk_id: str) -> Dict[str, Any]:
        return self._tools.fetch_chunk(chunk_id=chunk_id)

    def expand_span(self, doc_id: str, start: int, end: int, radius: int = 200) -> Dict[str, Any]:
        return self._tools.expand_span(doc_id=doc_id, start=start, end=end, radius=radius)
