"""
Comprehensive tests for crewai-soul.

Run with: python -m pytest tests/ -v
Or standalone: python tests/test_memory.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from crewai_soul import SoulMemory
from crewai_soul.memory import MemoryMatch, MemoryRecord


class TestSoulMemory:
    """Test suite for SoulMemory."""
    
    def setup_method(self):
        """Create a temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.soul_path = os.path.join(self.temp_dir, "SOUL.md")
        self.memory_path = os.path.join(self.temp_dir, "MEMORY.md")
    
    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_files(self):
        """Test that SoulMemory creates SOUL.md and MEMORY.md."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        assert os.path.exists(self.soul_path), "SOUL.md should be created"
        assert os.path.exists(self.memory_path), "MEMORY.md should be created"
    
    def test_init_no_auto_create(self):
        """Test that auto_create=False doesn't create files."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
            auto_create=False,
        )
        
        assert not os.path.exists(self.soul_path)
        assert not os.path.exists(self.memory_path)
    
    def test_remember_basic(self):
        """Test basic remember functionality."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("We decided to use PostgreSQL for the database.")
        
        content = Path(self.memory_path).read_text()
        assert "PostgreSQL" in content
        assert "## " in content  # Has timestamp header
    
    def test_remember_with_scope(self):
        """Test remember with scope tag."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("API rate limit is 1000/min", scope="/config/api")
        
        content = Path(self.memory_path).read_text()
        assert "/config/api" in content
        assert "**Scope:**" in content
    
    def test_remember_with_metadata(self):
        """Test remember with metadata."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Important decision", metadata={"priority": "high"})
        
        content = Path(self.memory_path).read_text()
        assert "priority" in content
    
    def test_recall_finds_keyword(self):
        """Test that recall finds entries by keyword."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("We decided to use PostgreSQL for the database.")
        memory.remember("The API rate limit is 1000 requests per minute.")
        memory.remember("Sprint velocity is 42 story points.")
        
        matches = memory.recall("database")
        assert len(matches) >= 1
        assert any("PostgreSQL" in m.content for m in matches)
    
    def test_recall_scores_correctly(self):
        """Test that recall scores multiple keyword matches higher."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("The database uses PostgreSQL.")
        memory.remember("PostgreSQL database configuration is in config.yaml.")
        
        matches = memory.recall("PostgreSQL database")
        assert len(matches) >= 1
        
        # Entry with both words should score higher
        if len(matches) >= 2:
            assert matches[0].score >= matches[1].score
    
    def test_recall_case_insensitive(self):
        """Test that recall is case insensitive."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("We use POSTGRESQL for everything.")
        
        matches = memory.recall("postgresql")
        assert len(matches) >= 1
        
        matches = memory.recall("PostgreSQL")
        assert len(matches) >= 1
    
    def test_recall_with_scope_filter(self):
        """Test that recall filters by scope."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Database config", scope="/config")
        memory.remember("Database design", scope="/design")
        
        matches = memory.recall("database", scope="/config")
        assert len(matches) == 1
        assert "config" in matches[0].content.lower() or "/config" in str(matches[0].record.scope)
    
    def test_recall_respects_limit(self):
        """Test that recall respects the limit parameter."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        for i in range(10):
            memory.remember(f"Database entry number {i}")
        
        matches = memory.recall("database", limit=3)
        assert len(matches) <= 3
    
    def test_recall_empty_memory(self):
        """Test recall on empty memory returns empty list."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        matches = memory.recall("anything")
        assert matches == []
    
    def test_forget_all(self):
        """Test that forget() clears all memory."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Entry 1")
        memory.remember("Entry 2")
        
        memory.forget()
        
        content = Path(self.memory_path).read_text()
        assert "Entry 1" not in content
        assert "Entry 2" not in content
        
        matches = memory.recall("Entry")
        assert len(matches) == 0
    
    def test_forget_by_scope(self):
        """Test that forget(scope) only removes matching entries."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Keep this", scope="/keep")
        memory.remember("Delete this", scope="/delete")
        
        removed = memory.forget(scope="/delete")
        
        content = Path(self.memory_path).read_text()
        assert "Keep this" in content
        assert "Delete this" not in content
    
    def test_reset_clears_memory(self):
        """Test that reset() clears memory (alias for forget)."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Test entry")
        memory.reset()
        
        matches = memory.recall("Test")
        assert len(matches) == 0
    
    def test_info_returns_stats(self):
        """Test that info() returns correct statistics."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Entry 1")
        memory.remember("Entry 2")
        memory.remember("Entry 3")
        
        info = memory.info()
        
        assert info["total_entries"] == 3
        assert info["memory_size_bytes"] > 0
        assert "soul_path" in info
        assert "memory_path" in info
    
    def test_tree_shows_structure(self):
        """Test that tree() shows file structure."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        tree = memory.tree()
        
        assert "SOUL" in tree
        assert "MEMORY" in tree
        assert "✓" in tree  # Files exist
    
    def test_extract_memories(self):
        """Test fact extraction from text."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        text = "We decided to use PostgreSQL. The budget is $50k. Sarah will lead."
        facts = memory.extract_memories(text)
        
        assert len(facts) >= 2
        assert any("PostgreSQL" in f for f in facts)
    
    def test_scope_returns_self(self):
        """Test that scope() returns self for method chaining."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        scoped = memory.scope("/test")
        assert scoped is memory
    
    def test_save_alias(self):
        """Test that save() works as alias for remember()."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.save("Saved content")
        
        matches = memory.recall("Saved")
        assert len(matches) >= 1
    
    def test_search_alias(self):
        """Test that search() works as alias for recall()."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        memory.remember("Searchable content")
        
        matches = memory.search("Searchable")
        assert len(matches) >= 1
    
    def test_multiple_memories_same_timestamp(self):
        """Test handling multiple memories added quickly."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        for i in range(5):
            memory.remember(f"Quick entry {i}")
        
        info = memory.info()
        assert info["total_entries"] == 5
    
    def test_special_characters_in_content(self):
        """Test handling of special characters."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        special = "Code: `print('hello')` and **bold** and $100"
        memory.remember(special)
        
        matches = memory.recall("Code")
        assert len(matches) >= 1
        assert "print" in matches[0].content
    
    def test_unicode_content(self):
        """Test handling of unicode characters."""
        memory = SoulMemory(
            soul_path=self.soul_path,
            memory_path=self.memory_path,
        )
        
        unicode_text = "日本語テスト and émojis 🎉"
        memory.remember(unicode_text)
        
        content = Path(self.memory_path).read_text(encoding="utf-8")
        assert "日本語" in content
        assert "🎉" in content
    
    def test_memory_match_properties(self):
        """Test MemoryMatch dataclass properties."""
        record = MemoryRecord(
            content="Test content",
            timestamp="2026-03-06 12:00:00 UTC",
            scope="/test",
        )
        match = MemoryMatch(record=record, score=0.75)
        
        assert match.content == "Test content"
        assert match.timestamp == "2026-03-06 12:00:00 UTC"
        assert match.score == 0.75


def run_tests():
    """Run all tests and print results."""
    test = TestSoulMemory()
    passed = 0
    failed = 0
    
    methods = [m for m in dir(test) if m.startswith("test_")]
    
    print(f"\n{'='*60}")
    print("crewai-soul test suite")
    print(f"{'='*60}\n")
    
    for method_name in sorted(methods):
        test.setup_method()
        try:
            getattr(test, method_name)()
            print(f"✅ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {method_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {method_name}: {type(e).__name__}: {e}")
            failed += 1
        finally:
            test.teardown_method()
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
