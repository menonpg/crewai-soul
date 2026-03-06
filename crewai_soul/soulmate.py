"""
SoulMate API Client: Enterprise memory backend integration.

SoulMate is the enterprise/cloud version of soul-agent, providing:
- Multi-tenant memory isolation
- Advanced RAG with hybrid retrieval
- Team collaboration features
- Production-grade infrastructure

API: https://soulmate-api-production.up.railway.app
Docs: https://menonpg.github.io/soulmate
"""

import os
from typing import Any, Dict, List, Optional

import httpx


class SoulMateClient:
    """
    Client for SoulMate API - enterprise memory backend.
    
    Usage:
        from crewai_soul import SoulMateClient
        
        client = SoulMateClient(api_key="your-key")
        
        # Store memory
        client.remember("Important decision made today", scope="/project/alpha")
        
        # Recall memories
        results = client.recall("What decisions were made?")
        
        # Ask with full RAG+RLM
        answer = client.ask("Summarize our project progress")
    """
    
    DEFAULT_URL = "https://soulmate-api-production.up.railway.app"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize SoulMate client.
        
        Args:
            api_key: API key (or SOULMATE_API_KEY env var)
            base_url: API base URL (or SOULMATE_URL env var)
            tenant_id: Tenant ID for multi-tenant isolation
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("SOULMATE_API_KEY")
        self.base_url = (base_url or os.environ.get("SOULMATE_URL") or self.DEFAULT_URL).rstrip("/")
        self.tenant_id = tenant_id or os.environ.get("SOULMATE_TENANT_ID")
        self.timeout = timeout
        
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._build_headers(),
        )
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers
    
    def health(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    def remember(
        self,
        content: str,
        scope: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a memory in SoulMate.
        
        Args:
            content: The content to remember
            scope: Optional scope path (e.g., "/project/alpha")
            metadata: Optional metadata dict
            
        Returns:
            Response with memory ID
        """
        payload = {"content": content}
        if scope:
            payload["scope"] = scope
        if metadata:
            payload["metadata"] = metadata
        
        response = self._client.post("/api/memory", json=payload)
        response.raise_for_status()
        return response.json()
    
    def recall(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search memories in SoulMate.
        
        Args:
            query: Search query
            limit: Maximum results
            scope: Optional scope filter
            min_score: Minimum relevance score
            
        Returns:
            List of memory matches with scores
        """
        params = {"q": query, "limit": limit}
        if scope:
            params["scope"] = scope
        if min_score > 0:
            params["min_score"] = min_score
        
        response = self._client.get("/api/memory/search", params=params)
        response.raise_for_status()
        return response.json()
    
    def ask(
        self,
        query: str,
        scope: Optional[str] = None,
        provider: str = "anthropic",
    ) -> Dict[str, Any]:
        """
        Ask a question with RAG+RLM retrieval.
        
        Args:
            query: The question to ask
            scope: Optional scope filter
            provider: LLM provider to use
            
        Returns:
            Response with answer, route, and context
        """
        payload = {"query": query, "provider": provider}
        if scope:
            payload["scope"] = scope
        
        response = self._client.post("/api/ask", json=payload)
        response.raise_for_status()
        return response.json()
    
    def forget(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear memories.
        
        Args:
            scope: Optional scope to clear (all if None)
            
        Returns:
            Response with count of deleted memories
        """
        params = {}
        if scope:
            params["scope"] = scope
        
        response = self._client.delete("/api/memory", params=params)
        response.raise_for_status()
        return response.json()
    
    def info(self) -> Dict[str, Any]:
        """Get memory statistics."""
        response = self._client.get("/api/memory/info")
        response.raise_for_status()
        return response.json()
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "SoulMateClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


# Convenience function
def connect(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> SoulMateClient:
    """
    Connect to SoulMate API.
    
    Usage:
        from crewai_soul.soulmate import connect
        
        with connect() as soulmate:
            soulmate.remember("Important fact")
            results = soulmate.recall("fact")
    """
    return SoulMateClient(api_key=api_key, base_url=base_url)


__all__ = ["SoulMateClient", "connect"]
