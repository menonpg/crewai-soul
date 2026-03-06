"""
SoulMemory: Markdown-native memory backend for CrewAI.

Two modes:
1. Standalone (no deps): Basic keyword matching on MEMORY.md
2. With soul-agent: Full RAG + RLM hybrid retrieval, multi-provider support

Install with RAG support:
    pip install crewai-soul[rag]
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field


# ── Try to import soul-agent for advanced features ────────────────────────────

_SOUL_AGENT_AVAILABLE = False
_HybridAgent = None

try:
    from hybrid_agent import HybridAgent as _HybridAgent
    _SOUL_AGENT_AVAILABLE = True
except ImportError:
    try:
        from soul import Agent as _HybridAgent
        _SOUL_AGENT_AVAILABLE = True
    except ImportError:
        pass


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """A single memory entry."""
    content: str
    timestamp: str = ""
    scope: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MemoryMatch:
    """A memory recall result with score."""
    record: MemoryRecord
    score: float
    
    @property
    def content(self) -> str:
        return self.record.content
    
    @property
    def timestamp(self) -> str:
        return self.record.timestamp


# ── SoulMemory ────────────────────────────────────────────────────────────────

class SoulMemory:
    """
    Markdown-native memory for CrewAI agents.
    
    Drop-in replacement for CrewAI's built-in Memory class.
    Uses SOUL.md for identity and MEMORY.md for persistent memory.
    
    Basic usage (standalone, no extra deps):
        from crewai_soul import SoulMemory
        memory = SoulMemory()
        memory.remember("We decided to use PostgreSQL.")
        matches = memory.recall("database")
    
    With soul-agent (RAG + RLM):
        pip install crewai-soul[rag]
        
        memory = SoulMemory(
            provider="anthropic",
            use_hybrid=True,
        )
    
    With CrewAI:
        from crewai import Crew
        from crewai_soul import SoulMemory
        
        crew = Crew(
            agents=[...],
            tasks=[...],
            memory=SoulMemory(),
        )
    """
    
    DEFAULT_SOUL = """# Soul

You are a helpful AI assistant working as part of a CrewAI crew.
You remember past interactions and build on previous context.
Be concise, accurate, and helpful.
"""
    
    def __init__(
        self,
        soul_path: str = "SOUL.md",
        memory_path: str = "MEMORY.md",
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        use_hybrid: bool = True,  # Default True since soul-agent is now required
        auto_create: bool = True,
        # Advanced options for soul-agent
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        azure_embedding_endpoint: Optional[str] = None,
        azure_embedding_key: Optional[str] = None,
    ):
        """
        Initialize SoulMemory.
        
        Args:
            soul_path: Path to SOUL.md (agent identity)
            memory_path: Path to MEMORY.md (memory log)
            provider: LLM provider ("anthropic", "openai", "gemini")
            api_key: API key for provider (or use env var)
            use_hybrid: Enable RAG+RLM via soul-agent (requires soul-agent)
            auto_create: Create files if they don't exist
            qdrant_url: Qdrant URL for vector search
            qdrant_api_key: Qdrant API key
            azure_embedding_endpoint: Azure OpenAI embeddings endpoint
            azure_embedding_key: Azure OpenAI embeddings key
        """
        self.soul_path = Path(soul_path)
        self.memory_path = Path(memory_path)
        self.provider = provider
        self.api_key = api_key
        self.use_hybrid = use_hybrid and _SOUL_AGENT_AVAILABLE
        
        self._hybrid_agent = None
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._azure_embedding_endpoint = azure_embedding_endpoint
        self._azure_embedding_key = azure_embedding_key
        
        if auto_create:
            self._ensure_files()
        
        # Initialize hybrid agent if requested and available
        if self.use_hybrid:
            self._init_hybrid_agent()
    
    def _ensure_files(self) -> None:
        """Create SOUL.md and MEMORY.md if they don't exist."""
        if not self.soul_path.exists():
            self.soul_path.write_text(self.DEFAULT_SOUL)
        
        if not self.memory_path.exists():
            self.memory_path.write_text("# Memory Log\n\n")
    
    def _init_hybrid_agent(self) -> None:
        """Initialize soul-agent's HybridAgent for advanced retrieval."""
        if not _SOUL_AGENT_AVAILABLE or not _HybridAgent:
            return
        
        try:
            self._hybrid_agent = _HybridAgent(
                soul_path=str(self.soul_path),
                memory_path=str(self.memory_path),
                provider=self.provider,
                api_key=self.api_key,
                qdrant_url=self._qdrant_url,
                qdrant_api_key=self._qdrant_api_key,
                azure_embedding_endpoint=self._azure_embedding_endpoint,
                azure_embedding_key=self._azure_embedding_key,
            )
        except Exception as e:
            # Fall back to basic mode
            self._hybrid_agent = None
            self.use_hybrid = False
    
    def remember(
        self,
        content: str,
        scope: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a memory. Appends to MEMORY.md with timestamp.
        
        Args:
            content: The content to remember
            scope: Optional scope tag (e.g., "/project/alpha")
            metadata: Optional metadata dict
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        entry = f"\n## {timestamp}\n"
        if scope:
            entry += f"**Scope:** `{scope}`\n\n"
        entry += f"{content}\n"
        
        if metadata:
            entry += f"\n*Metadata: {metadata}*\n"
        
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(entry)
        
        # If using hybrid agent, also update its index
        if self._hybrid_agent and hasattr(self._hybrid_agent, 'remember'):
            try:
                self._hybrid_agent.remember(content)
            except Exception:
                pass  # Non-critical, file is already updated
    
    def recall(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
    ) -> List[MemoryMatch]:
        """
        Retrieve relevant memories.
        
        If soul-agent is available and use_hybrid=True, uses RAG/RLM.
        Otherwise, uses keyword matching.
        
        Args:
            query: What to search for
            limit: Maximum results to return
            scope: Optional scope filter
            
        Returns:
            List of MemoryMatch objects sorted by relevance
        """
        if not self.memory_path.exists():
            return []
        
        # Try hybrid retrieval first
        if self._hybrid_agent:
            try:
                return self._hybrid_recall(query, limit, scope)
            except Exception:
                pass  # Fall back to basic
        
        # Basic keyword matching
        return self._basic_recall(query, limit, scope)
    
    def _hybrid_recall(
        self,
        query: str,
        limit: int,
        scope: Optional[str],
    ) -> List[MemoryMatch]:
        """Use soul-agent's RAG/RLM for retrieval."""
        result = self._hybrid_agent.ask(query, remember=False)
        
        # Parse context from RAG result
        context = result.get("rag_context", "") or ""
        
        # Create a single high-relevance match from the context
        if context:
            record = MemoryRecord(
                content=context[:500],  # Truncate for display
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            )
            return [MemoryMatch(record=record, score=1.0)]
        
        return []
    
    def _basic_recall(
        self,
        query: str,
        limit: int,
        scope: Optional[str],
    ) -> List[MemoryMatch]:
        """Basic keyword matching for recall."""
        content = self.memory_path.read_text(encoding="utf-8")
        entries = self._parse_entries(content)
        
        # Filter by scope if provided
        if scope:
            entries = [e for e in entries if scope in e.get("scope", "")]
        
        # Tokenize helper
        def tokenize(text: str) -> set:
            return set(re.findall(r'\b\w+\b', text.lower()))
        
        query_words = tokenize(query)
        if not query_words:
            return []
        
        scored = []
        for entry in entries:
            entry_words = tokenize(entry["content"])
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / len(query_words)
                record = MemoryRecord(
                    content=entry["content"],
                    timestamp=entry.get("timestamp", ""),
                    scope=entry.get("scope", ""),
                )
                scored.append(MemoryMatch(record=record, score=score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        
        return scored[:limit]
    
    def _parse_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse MEMORY.md into structured entries."""
        entries = []
        current_entry = None
        
        for line in content.split("\n"):
            if line.startswith("## "):
                if current_entry and current_entry.get("content", "").strip():
                    entries.append(current_entry)
                current_entry = {
                    "timestamp": line[3:].strip(),
                    "content": "",
                    "scope": "",
                }
            elif current_entry is not None:
                if line.startswith("**Scope:**"):
                    # Extract scope from backticks
                    match = re.search(r'`([^`]+)`', line)
                    if match:
                        current_entry["scope"] = match.group(1)
                elif not line.startswith("*Metadata:"):
                    current_entry["content"] += line + "\n"
        
        if current_entry and current_entry.get("content", "").strip():
            entries.append(current_entry)
        
        # Clean up content
        for entry in entries:
            entry["content"] = entry["content"].strip()
        
        return entries
    
    def forget(self, scope: Optional[str] = None) -> int:
        """
        Clear memories. If scope provided, only clear that scope.
        
        Args:
            scope: Optional scope to clear (clears all if None)
            
        Returns:
            Number of entries removed (-1 if unknown/all)
        """
        if not self.memory_path.exists():
            return 0
        
        if scope is None:
            # Clear everything
            self.memory_path.write_text("# Memory Log\n\n", encoding="utf-8")
            return -1
        
        # Filter out entries matching scope
        content = self.memory_path.read_text(encoding="utf-8")
        entries = self._parse_entries(content)
        
        original_count = len(entries)
        entries = [e for e in entries if scope not in e.get("scope", "")]
        
        # Rewrite file
        new_content = "# Memory Log\n"
        for entry in entries:
            new_content += f"\n## {entry['timestamp']}\n"
            if entry.get("scope"):
                new_content += f"**Scope:** `{entry['scope']}`\n\n"
            new_content += f"{entry['content']}\n"
        
        self.memory_path.write_text(new_content, encoding="utf-8")
        
        return original_count - len(entries)
    
    def reset(self) -> None:
        """Clear all memory. Alias for forget()."""
        self.forget()
    
    def tree(self) -> str:
        """Return the memory structure (file paths)."""
        soul_exists = "✓" if self.soul_path.exists() else "✗"
        mem_exists = "✓" if self.memory_path.exists() else "✗"
        hybrid = "✓ (RAG+RLM)" if self._hybrid_agent else "✗ (basic)"
        
        return (
            f"SoulMemory\n"
            f"├── SOUL:   {self.soul_path} [{soul_exists}]\n"
            f"├── MEMORY: {self.memory_path} [{mem_exists}]\n"
            f"└── Hybrid: {hybrid}"
        )
    
    def info(self, scope: str = "/") -> Dict[str, Any]:
        """Return memory statistics."""
        entries = []
        size_bytes = 0
        
        if self.memory_path.exists():
            content = self.memory_path.read_text(encoding="utf-8")
            entries = self._parse_entries(content)
            size_bytes = len(content.encode("utf-8"))
        
        # Filter by scope if not root
        if scope != "/":
            entries = [e for e in entries if scope in e.get("scope", "")]
        
        return {
            "total_entries": len(entries),
            "soul_path": str(self.soul_path),
            "memory_path": str(self.memory_path),
            "memory_size_bytes": size_bytes,
            "hybrid_enabled": self._hybrid_agent is not None,
            "provider": self.provider,
        }
    
    def extract_memories(self, text: str) -> List[str]:
        """
        Extract atomic facts from text.
        Splits by sentence boundaries.
        
        Args:
            text: Input text to extract facts from
            
        Returns:
            List of atomic fact strings
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences
        facts = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        return facts
    
    def scope(self, path: str) -> "SoulMemory":
        """
        Return a scoped view of memory.
        
        For full compatibility, returns self but could be extended
        to create a filtered view.
        
        Args:
            path: Scope path (e.g., "/agent/researcher")
            
        Returns:
            Self (scoped filtering happens in recall)
        """
        # In a more advanced implementation, this could return
        # a ScopedSoulMemory wrapper that filters all operations
        return self
    
    # ── CrewAI compatibility methods ──────────────────────────────────────────
    
    def save(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Alias for remember() for CrewAI compatibility."""
        self.remember(content, metadata=metadata)
    
    def search(self, query: str, limit: int = 5) -> List[MemoryMatch]:
        """Alias for recall() for CrewAI compatibility."""
        return self.recall(query, limit=limit)


# ── Exports ───────────────────────────────────────────────────────────────────

__all__ = ["SoulMemory", "MemoryMatch", "MemoryRecord"]
