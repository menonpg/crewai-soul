"""
SoulMemory: Markdown-native memory backend for CrewAI.

Instead of vector databases and LLM-inferred scopes, SoulMemory uses
two simple markdown files:
- SOUL.md: Agent identity (who it is, how it behaves)
- MEMORY.md: Timestamped log of all interactions

Human-readable. Git-versionable. No infrastructure required.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any
from dataclasses import dataclass


@dataclass
class MemoryMatch:
    """A memory recall result."""
    content: str
    score: float
    timestamp: str
    
    @property
    def record(self):
        """Compatibility with CrewAI's memory interface."""
        return self


class SoulMemory:
    """
    Markdown-native memory for CrewAI agents.
    
    Drop-in replacement for CrewAI's built-in Memory class.
    Uses SOUL.md for identity and MEMORY.md for persistent memory.
    
    Example:
        from crewai import Crew, Agent, Task
        from crewai_soul import SoulMemory
        
        memory = SoulMemory(memory_path="crew_memory.md")
        
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            memory=memory,
        )
    """
    
    def __init__(
        self,
        soul_path: str = "SOUL.md",
        memory_path: str = "MEMORY.md",
        provider: str = "anthropic",
        auto_create: bool = True,
    ):
        """
        Initialize SoulMemory.
        
        Args:
            soul_path: Path to SOUL.md (agent identity)
            memory_path: Path to MEMORY.md (memory log)
            provider: LLM provider for retrieval ("anthropic", "openai", "gemini")
            auto_create: Create files if they don't exist
        """
        self.soul_path = Path(soul_path)
        self.memory_path = Path(memory_path)
        self.provider = provider
        
        if auto_create:
            self._ensure_files()
    
    def _ensure_files(self):
        """Create SOUL.md and MEMORY.md if they don't exist."""
        if not self.soul_path.exists():
            self.soul_path.write_text(
                "# Soul\n\n"
                "You are a helpful AI assistant working as part of a CrewAI crew.\n"
                "You remember past interactions and build on previous context.\n"
            )
        
        if not self.memory_path.exists():
            self.memory_path.write_text("# Memory Log\n\n")
    
    def remember(
        self,
        content: str,
        scope: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a memory. Appends to MEMORY.md with timestamp.
        
        Args:
            content: The content to remember
            scope: Optional scope (stored as a tag, but we don't use hierarchies)
            metadata: Optional metadata dict
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        entry = f"\n## {timestamp}\n"
        if scope:
            entry += f"**Scope:** `{scope}`\n\n"
        entry += f"{content}\n"
        
        if metadata:
            entry += f"\n*Metadata: {metadata}*\n"
        
        with open(self.memory_path, "a") as f:
            f.write(entry)
    
    def recall(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
    ) -> List[MemoryMatch]:
        """
        Retrieve relevant memories.
        
        Uses simple keyword matching for now. For semantic search,
        install soul-agent[rag] and we'll use the RAG backend.
        
        Args:
            query: What to search for
            limit: Maximum results to return
            scope: Optional scope filter
            
        Returns:
            List of MemoryMatch objects
        """
        if not self.memory_path.exists():
            return []
        
        content = self.memory_path.read_text()
        
        # Parse memory entries
        entries = self._parse_entries(content)
        
        # Filter by scope if provided
        if scope:
            entries = [e for e in entries if scope in e.get("scope", "")]
        
        # Score by keyword overlap (simple but effective)
        import re
        def tokenize(text):
            return set(re.findall(r'\b\w+\b', text.lower()))
        
        query_words = tokenize(query)
        scored = []
        
        for entry in entries:
            entry_words = tokenize(entry["content"])
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, entry))
        
        # Sort by score, take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, entry in scored[:limit]:
            results.append(MemoryMatch(
                content=entry["content"],
                score=score,
                timestamp=entry.get("timestamp", ""),
            ))
        
        return results
    
    def _parse_entries(self, content: str) -> List[dict]:
        """Parse MEMORY.md into structured entries."""
        entries = []
        current_entry = None
        
        for line in content.split("\n"):
            if line.startswith("## "):
                if current_entry:
                    entries.append(current_entry)
                current_entry = {
                    "timestamp": line[3:].strip(),
                    "content": "",
                    "scope": "",
                }
            elif current_entry:
                if line.startswith("**Scope:**"):
                    current_entry["scope"] = line.split("`")[1] if "`" in line else ""
                else:
                    current_entry["content"] += line + "\n"
        
        if current_entry:
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
            Number of entries removed
        """
        if scope is None:
            # Clear everything
            self.memory_path.write_text("# Memory Log\n\n")
            return -1  # Unknown count
        
        # Filter out entries matching scope
        content = self.memory_path.read_text()
        entries = self._parse_entries(content)
        
        original_count = len(entries)
        entries = [e for e in entries if scope not in e.get("scope", "")]
        
        # Rewrite file
        new_content = "# Memory Log\n\n"
        for entry in entries:
            new_content += f"\n## {entry['timestamp']}\n"
            if entry.get("scope"):
                new_content += f"**Scope:** `{entry['scope']}`\n\n"
            new_content += f"{entry['content']}\n"
        
        self.memory_path.write_text(new_content)
        
        return original_count - len(entries)
    
    def tree(self) -> str:
        """
        Return the memory structure.
        For SoulMemory, this is just the file paths.
        """
        return f"SOUL: {self.soul_path}\nMEMORY: {self.memory_path}"
    
    def info(self, scope: str = "/") -> dict:
        """Return memory statistics."""
        entries = []
        if self.memory_path.exists():
            entries = self._parse_entries(self.memory_path.read_text())
        
        return {
            "total_entries": len(entries),
            "soul_path": str(self.soul_path),
            "memory_path": str(self.memory_path),
            "memory_size_bytes": self.memory_path.stat().st_size if self.memory_path.exists() else 0,
        }
    
    def extract_memories(self, text: str) -> List[str]:
        """
        Extract atomic facts from text.
        Simple implementation: split by sentences.
        """
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # Aliases for CrewAI compatibility
    def scope(self, path: str) -> "SoulMemory":
        """Return a scoped view (for compatibility, just returns self with scope noted)."""
        # In a full implementation, this would filter recall() by scope
        return self
    
    def reset(self) -> None:
        """Clear all memory."""
        self.forget()
