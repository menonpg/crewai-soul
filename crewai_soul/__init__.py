"""
crewai-soul: Markdown-native memory for CrewAI agents.

Human-readable, git-versionable, no database required.

Basic usage:
    from crewai_soul import SoulMemory
    
    memory = SoulMemory()
    memory.remember("We decided to use PostgreSQL.")
    matches = memory.recall("database")

With CrewAI:
    from crewai import Crew
    from crewai_soul import SoulMemory
    
    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=SoulMemory(),
    )

With soul-agent (RAG + RLM):
    pip install crewai-soul[rag]
    
    memory = SoulMemory(provider="anthropic", use_hybrid=True)
"""

from .memory import SoulMemory, MemoryMatch, MemoryRecord

__version__ = "0.2.0"
__all__ = ["SoulMemory", "MemoryMatch", "MemoryRecord"]
