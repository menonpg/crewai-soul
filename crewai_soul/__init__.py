"""
crewai-soul: The soul ecosystem for CrewAI agents.

The complete memory, identity, and intelligence stack:
- **soul-agent**: RAG + RLM hybrid memory with persistent identity
- **soul-schema**: Auto-generate database semantic layers
- **SoulMate**: Enterprise memory API integration

Install:
    pip install crewai-soul

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

Database Schema Intelligence:
    from crewai_soul import SchemaMemory
    
    schema = SchemaMemory("postgresql://user:pass@host/db")
    schema.generate()
    context = schema.context_for("Show me revenue by region")

Enterprise (SoulMate API):
    from crewai_soul import SoulMateClient
    
    client = SoulMateClient(api_key="...")
    client.remember("Important decision")
    results = client.recall("decision")
"""

from .memory import SoulMemory, MemoryMatch, MemoryRecord

__version__ = "0.3.0"

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
    # Core memory
    "SoulMemory",
    "MemoryMatch", 
    "MemoryRecord",
    # Enterprise API (lazy loaded)
    "SoulMateClient",
    "soulmate_connect",
    # Database schema (lazy loaded)
    "SchemaMemory",
]
