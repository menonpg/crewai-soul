"""
crewai-soul: The soul ecosystem for CrewAI agents.

Two deployment options — same great memory:

## Option 1: Local (file-based)
    from crewai_soul import SoulMemory
    
    memory = SoulMemory()  # Uses SOUL.md + MEMORY.md
    memory.remember("We decided to use PostgreSQL.")
    matches = memory.recall("database")

## Option 2: SoulMate (managed cloud) ⭐ Recommended for production
    from crewai_soul import SoulMateMemory
    
    memory = SoulMateMemory(api_key="...")  # We handle infrastructure
    memory.remember("We decided to use PostgreSQL.")
    matches = memory.recall("database")

Both use the same soul-agent RAG+RLM under the hood.

## With CrewAI
    from crewai import Crew
    from crewai_soul import SoulMemory, SoulMateMemory
    
    # Local
    crew = Crew(agents=[...], memory=SoulMemory())
    
    # Or managed
    crew = Crew(agents=[...], memory=SoulMateMemory(api_key="..."))

## Factory function
    from crewai_soul import create_memory
    
    memory = create_memory("local")     # File-based
    memory = create_memory("soulmate")  # Managed cloud

## Database Schema Intelligence
    from crewai_soul import SchemaMemory
    
    schema = SchemaMemory("postgresql://...")
    context = schema.context_for("Show me revenue by region")

Get SoulMate API key: https://menonpg.github.io/soulmate
"""

from .memory import SoulMemory, SoulMateMemory, MemoryMatch, MemoryRecord, create_memory

__version__ = "0.3.1"

# Lazy imports for optional integrations
def __getattr__(name):
    if name == "SoulMateClient":
        from .soulmate import SoulMateClient
        return SoulMateClient
    elif name == "soulmate_connect":
        from .soulmate import connect
        return connect
    elif name == "SchemaMemory":
        from .schema import SchemaMemory
        return SchemaMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Memory backends
    "SoulMemory",        # Local file-based
    "SoulMateMemory",    # Managed cloud (SoulMate API)
    "create_memory",     # Factory function
    "MemoryMatch", 
    "MemoryRecord",
    # SoulMate API client (lazy loaded)
    "SoulMateClient",
    "soulmate_connect",
    # Database schema (lazy loaded)
    "SchemaMemory",
]
