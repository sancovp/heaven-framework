#!/usr/bin/env python3
"""
RedactionManager: Manages redaction rules for SEED v0 Publishing Platform

This module handles loading, saving, and applying redaction rules from redacted.json.
Redaction is done through exact string matching and replacement with [REDACTED].

VENDORED into heaven-framework 2026-07-08 (faithful copy). Canonical source:
application/seed-mcp/src/seed_mcp/publishing/redaction_manager.py (seed-mcp / seed-publishing).
It was previously imported via a hardcoded host path (/home/GOD/seed_v0_publishing) in
redaction_tool.py, which does not exist in-container -> ModuleNotFoundError every load.
Self-contained (stdlib only), so vendored rather than lazy-imported. If the canonical source
changes, re-copy it here. See redaction_tool.py's relative import.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedactionManager:
    """Manages redaction rules and applies them to content."""
    
    def __init__(self, redacted_file_path: str = "redacted.json"):
        """
        Initialize RedactionManager with path to redacted.json file.
        
        Args:
            redacted_file_path: Path to the redacted.json file
        """
        self.redacted_file_path = redacted_file_path
        self.rules: Dict[str, str] = {}
        self.load_rules()
    
    def load_rules(self) -> bool:
        """
        Load redaction rules from redacted.json file.
        
        Returns:
            True if rules loaded successfully, False otherwise
        """
        if not os.path.exists(self.redacted_file_path):
            logger.info(f"No redacted.json found at {self.redacted_file_path}, creating empty rules file")
            self.save_rules()
            return True
        
        try:
            with open(self.redacted_file_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            logger.info(f"Loaded {len(self.rules)} redaction rules from {self.redacted_file_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse redacted.json: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load redacted.json: {e}")
            return False
    
    def save_rules(self) -> bool:
        """
        Save current redaction rules to redacted.json file.
        
        Returns:
            True if rules saved successfully, False otherwise
        """
        try:
            with open(self.redacted_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.rules, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.rules)} redaction rules to {self.redacted_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save redacted.json: {e}")
            return False
    
    def add_rule(self, sensitive_term: str, replacement: str = "[REDACTED]") -> bool:
        """
        Add a new redaction rule.
        
        Args:
            sensitive_term: The term to redact
            replacement: What to replace it with (default: [REDACTED])
        
        Returns:
            True if rule added successfully, False otherwise
        """
        if not sensitive_term:
            logger.error("Cannot add empty redaction rule")
            return False
        
        self.rules[sensitive_term] = replacement
        return self.save_rules()
    
    def remove_rule(self, sensitive_term: str) -> bool:
        """
        Remove a redaction rule.
        
        Args:
            sensitive_term: The term to stop redacting
        
        Returns:
            True if rule removed successfully, False if not found
        """
        if sensitive_term not in self.rules:
            logger.warning(f"Term '{sensitive_term}' not found in redaction rules")
            return False
        
        del self.rules[sensitive_term]
        return self.save_rules()
    
    def get_rules(self) -> Dict[str, str]:
        """
        Get all current redaction rules.
        
        Returns:
            Dictionary of redaction rules
        """
        return self.rules.copy()
    
    def _get_sorted_rules(self) -> List[Tuple[str, str]]:
        """Get rules sorted by length (longest first) to handle overlapping terms."""
        return sorted(self.rules.items(), key=lambda x: len(x[0]), reverse=True)
    
    def _apply_single_rule(self, content: str, sensitive_term: str, replacement: str) -> Tuple[str, int]:
        """Apply a single redaction rule to content."""
        count = content.count(sensitive_term)
        if count > 0:
            content = content.replace(sensitive_term, replacement)
            logger.debug(f"Redacted {count} occurrences of '{sensitive_term}'")
        return content, count
    
    def apply_redactions(self, content: str) -> Tuple[str, int]:
        """
        Apply all redaction rules to the given content.
        
        Args:
            content: The text content to redact
        
        Returns:
            Tuple of (redacted_content, redaction_count)
        """
        redacted_content = content
        total_redactions = 0
        
        for sensitive_term, replacement in self._get_sorted_rules():
            redacted_content, count = self._apply_single_rule(
                redacted_content, sensitive_term, replacement
            )
            total_redactions += count
        
        return redacted_content, total_redactions
    
    def apply_redactions_to_file(self, file_path: str) -> Tuple[str, int]:
        """
        Apply redactions to a file's content.
        
        Args:
            file_path: Path to the file to redact
        
        Returns:
            Tuple of (redacted_content, redaction_count)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.apply_redactions(content)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return "", 0
    
    def _should_process_file(self, file_path: Path, file_extensions: List[str]) -> bool:
        """Check if a file should be processed based on extension."""
        return file_path.is_file() and file_path.suffix in file_extensions
    
    def _prepare_target_file(self, source_path: Path, target_path: Path, file_path: Path) -> Path:
        """Prepare target file path and create necessary directories."""
        relative_path = file_path.relative_to(source_path)
        target_file = target_path / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        return target_file
    
    def _save_redacted_file(self, target_file: Path, content: str) -> bool:
        """Save redacted content to target file."""
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to save redacted file {target_file}: {e}", exc_info=True)
            return False
    
    def _process_single_file(self, file_path: Path, source_path: Path, 
                           target_path: Path) -> Tuple[str, int]:
        """Process a single file for redaction."""
        target_file = self._prepare_target_file(source_path, target_path, file_path)
        redacted_content, count = self.apply_redactions_to_file(str(file_path))
        relative_path = str(file_path.relative_to(source_path))
        
        if self._save_redacted_file(target_file, redacted_content):
            logger.info(f"Redacted {count} terms in {relative_path}")
            return relative_path, count
        else:
            return relative_path, -1
    
    def redact_directory(self, source_dir: str, target_dir: str, 
                        file_extensions: List[str] = ['.md', '.txt']) -> Dict[str, int]:
        """
        Apply redactions to all files in a directory and save to target directory.
        
        Args:
            source_dir: Source directory containing files to redact
            target_dir: Target directory to save redacted files
            file_extensions: List of file extensions to process
        
        Returns:
            Dictionary mapping file paths to redaction counts
        """
        results = {}
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in source_path.rglob('*'):
            if not self._should_process_file(file_path, file_extensions):
                continue
            
            relative_path, count = self._process_single_file(
                file_path, source_path, target_path
            )
            results[relative_path] = count
        
        return results
    
    def _find_term_occurrences(self, content: str, sensitive_term: str) -> List[int]:
        """Find all positions of a term in content."""
        positions = []
        index = 0
        while True:
            index = content.find(sensitive_term, index)
            if index == -1:
                break
            positions.append(index)
            index += len(sensitive_term)
        return positions
    
    def _create_preview_context(self, content: str, term: str, replacement: str, 
                               position: int, context_chars: int) -> Dict[str, str]:
        """Create a preview context for a single redaction."""
        start = max(0, position - context_chars)
        end = min(len(content), position + len(term) + context_chars)
        context = content[start:end]
        highlighted = context.replace(term, f"**{replacement}**")
        
        return {
            "term": term,
            "replacement": replacement,
            "context": highlighted,
            "position": position
        }
    
    def preview_redactions(self, content: str, context_chars: int = 30) -> List[Dict[str, str]]:
        """
        Preview what would be redacted without actually redacting.
        
        Args:
            content: The text content to preview
            context_chars: Number of characters to show before/after match
        
        Returns:
            List of dictionaries with redaction previews
        """
        previews = []
        
        for sensitive_term, replacement in self.rules.items():
            positions = self._find_term_occurrences(content, sensitive_term)
            for position in positions:
                preview = self._create_preview_context(
                    content, sensitive_term, replacement, position, context_chars
                )
                previews.append(preview)
        
        return previews


def _setup_test_manager() -> RedactionManager:
    """Create and configure a test RedactionManager instance."""
    manager = RedactionManager("test_redacted.json")
    print("Adding redaction rules...")
    manager.add_rule("SECRET_API_KEY_123", "[API_KEY_REDACTED]")
    manager.add_rule("john.doe@example.com", "[EMAIL_REDACTED]")
    manager.add_rule("/home/user/private", "[PATH_REDACTED]")
    return manager


def _get_test_content() -> str:
    """Return test content with sensitive information."""
    return """
    Here's some test content with sensitive information:
    - API Key: SECRET_API_KEY_123
    - Email: john.doe@example.com
    - Path: /home/user/private/data
    - Another mention of SECRET_API_KEY_123
    """


def _print_preview_results(manager: RedactionManager, content: str):
    """Print redaction preview results."""
    print("\nRedaction preview:")
    previews = manager.preview_redactions(content)
    for preview in previews:
        print(f"  - Will redact '{preview['term']}' at position {preview['position']}")


def _print_redaction_results(manager: RedactionManager, content: str):
    """Apply and print redaction results."""
    redacted, count = manager.apply_redactions(content)
    print(f"\nRedacted content ({count} redactions made):")
    print(redacted)


def main():
    """Test the RedactionManager functionality."""
    manager = _setup_test_manager()
    test_content = _get_test_content()
    
    print("\nOriginal content:")
    print(test_content)
    
    _print_preview_results(manager, test_content)
    _print_redaction_results(manager, test_content)
    
    # Clean up test file
    os.remove("test_redacted.json")
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()