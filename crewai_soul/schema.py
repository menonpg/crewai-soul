"""
Soul Schema Integration: Database semantic layer for AI agents.

soul-schema auto-generates semantic layers (column/table descriptions)
from any database using LLMs. This module integrates that capability
into CrewAI agents.

Install: pip install soul-schema (included with crewai-soul)
Repo: https://github.com/menonpg/soul-schema
"""

from typing import Any, Dict, List, Optional

# Import soul-schema
try:
    from soul_schema import SchemaGenerator, SchemaExporter
    from soul_schema.core import ColumnDescription, TableDescription
    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False
    SchemaGenerator = None
    SchemaExporter = None


class SchemaMemory:
    """
    Database schema memory for CrewAI agents.
    
    Allows agents to understand and query database structures
    using auto-generated semantic descriptions.
    
    Usage:
        from crewai_soul import SchemaMemory
        
        # Connect to database and generate descriptions
        schema = SchemaMemory("postgresql://user:pass@host/db")
        schema.generate()
        
        # Query schema knowledge
        result = schema.describe("customers")
        context = schema.context_for("Show me revenue by region")
    """
    
    def __init__(
        self,
        database_url: str,
        llm_provider: str = "anthropic",
        api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize SchemaMemory.
        
        Args:
            database_url: SQLAlchemy connection string
            llm_provider: LLM provider for generating descriptions
            api_key: API key (or use env var)
            cache_path: Path to cache generated schema
        """
        if not _SCHEMA_AVAILABLE:
            raise ImportError(
                "soul-schema not available. Install with: pip install soul-schema"
            )
        
        self.database_url = database_url
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.cache_path = cache_path
        
        self._generator = SchemaGenerator(
            database_url=database_url,
            llm_provider=llm_provider,
            api_key=api_key,
        )
        self._schema: Dict[str, TableDescription] = {}
        self._generated = False
    
    def generate(
        self,
        tables: Optional[List[str]] = None,
        sample_rows: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate semantic descriptions for database tables.
        
        Args:
            tables: Specific tables to describe (all if None)
            sample_rows: Number of sample rows to use for context
            
        Returns:
            Generated schema with descriptions
        """
        self._schema = self._generator.generate(
            tables=tables,
            sample_rows=sample_rows,
        )
        self._generated = True
        
        if self.cache_path:
            self.save(self.cache_path)
        
        return self._schema
    
    def describe(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get description for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table description with columns, or None if not found
        """
        if not self._generated:
            self.generate()
        
        table = self._schema.get(table_name)
        if not table:
            return None
        
        return {
            "name": table.name,
            "description": table.description,
            "columns": [
                {
                    "name": col.name,
                    "type": col.data_type,
                    "description": col.description,
                    "nullable": col.nullable,
                }
                for col in table.columns
            ],
        }
    
    def context_for(self, query: str, max_tables: int = 5) -> str:
        """
        Generate schema context for a natural language query.
        
        Args:
            query: Natural language query (e.g., "Show me revenue by region")
            max_tables: Maximum number of relevant tables to include
            
        Returns:
            Formatted schema context for LLM prompt injection
        """
        if not self._generated:
            self.generate()
        
        # Simple keyword matching for relevant tables
        query_lower = query.lower()
        scored_tables = []
        
        for table_name, table in self._schema.items():
            score = 0
            
            # Check table name
            if table_name.lower() in query_lower:
                score += 10
            
            # Check description
            if table.description and any(
                word in table.description.lower()
                for word in query_lower.split()
            ):
                score += 5
            
            # Check column names
            for col in table.columns:
                if col.name.lower() in query_lower:
                    score += 3
            
            if score > 0:
                scored_tables.append((score, table_name, table))
        
        # Sort by relevance
        scored_tables.sort(reverse=True, key=lambda x: x[0])
        
        # Build context
        context_parts = ["## Relevant Database Schema\n"]
        
        for _, table_name, table in scored_tables[:max_tables]:
            context_parts.append(f"\n### {table_name}")
            if table.description:
                context_parts.append(f"{table.description}\n")
            
            context_parts.append("| Column | Type | Description |")
            context_parts.append("|--------|------|-------------|")
            
            for col in table.columns:
                desc = col.description or ""
                context_parts.append(f"| {col.name} | {col.data_type} | {desc} |")
        
        return "\n".join(context_parts)
    
    def save(self, path: str, format: str = "json") -> None:
        """
        Save generated schema to file.
        
        Args:
            path: Output file path
            format: Export format (json, dbt, vanna)
        """
        if not _SCHEMA_AVAILABLE:
            raise ImportError("soul-schema required")
        
        exporter = SchemaExporter(self._schema)
        
        if format == "json":
            exporter.to_json(path)
        elif format == "dbt":
            exporter.to_dbt_yaml(path)
        elif format == "vanna":
            exporter.to_vanna(path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load(self, path: str) -> None:
        """
        Load schema from cached file.
        
        Args:
            path: Path to cached schema JSON
        """
        import json
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct schema objects
        # (simplified - actual implementation depends on soul-schema internals)
        self._schema = data
        self._generated = True
    
    def tables(self) -> List[str]:
        """List all known table names."""
        return list(self._schema.keys())
    
    def to_markdown(self) -> str:
        """Export full schema as markdown documentation."""
        if not self._generated:
            self.generate()
        
        lines = ["# Database Schema\n"]
        
        for table_name, table in self._schema.items():
            lines.append(f"\n## {table_name}")
            if hasattr(table, 'description') and table.description:
                lines.append(f"\n{table.description}\n")
            
            lines.append("\n| Column | Type | Description |")
            lines.append("|--------|------|-------------|")
            
            columns = table.columns if hasattr(table, 'columns') else []
            for col in columns:
                desc = col.description if hasattr(col, 'description') else ""
                dtype = col.data_type if hasattr(col, 'data_type') else "unknown"
                name = col.name if hasattr(col, 'name') else str(col)
                lines.append(f"| {name} | {dtype} | {desc} |")
        
        return "\n".join(lines)


# Check availability
def is_available() -> bool:
    """Check if soul-schema is available."""
    return _SCHEMA_AVAILABLE


__all__ = ["SchemaMemory", "is_available"]
