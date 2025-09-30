# redaction_tool.py

import os
import sys
import logging
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the SEED publishing directory to path so we can import redaction_manager
sys.path.insert(0, '/home/GOD/seed_v0_publishing')

try:
    from redaction_manager import RedactionManager
except ImportError as e:
    RedactionManager = None
    IMPORT_ERROR = str(e)
    logger.error(f"Failed to import RedactionManager: {e}", exc_info=True)

class RedactionToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'sensitive_term': {
            'name': 'sensitive_term',
            'type': 'str',
            'description': 'The exact sensitive string to redact (must match exactly as it appears in text)',
            'required': True
        },
        'replacement': {
            'name': 'replacement',
            'type': 'str',
            'description': 'Replacement text for the sensitive term (default: [REDACTED])',
            'required': False
        }
    }

def _validate_redaction_dependencies() -> tuple[bool, str]:
    """Validate that RedactionManager is available."""
    if RedactionManager is None:
        return False, f"❌ ERROR: Could not import RedactionManager: {IMPORT_ERROR}"
    return True, ""

def _validate_sensitive_term(sensitive_term: str) -> tuple[bool, str]:
    """Validate that sensitive term is not empty."""
    if not sensitive_term or not sensitive_term.strip():
        return False, "❌ ERROR: Sensitive term cannot be empty"
    return True, ""

def _create_success_message(sensitive_term: str, replacement: str) -> str:
    """Create success message for redaction rule addition."""
    return f"✅ SUCCESS: Added redaction rule\n" \
           f"Term: '{sensitive_term}'\n" \
           f"Replacement: '{replacement}'\n" \
           f"Rule saved to redacted.json"

def _add_redaction_rule_safely(sensitive_term: str, replacement: str) -> tuple[bool, str]:
    """Safely add redaction rule with error handling."""
    try:
        logger.info(f"Adding redaction rule for term: {sensitive_term[:20]}...")
        manager = RedactionManager()
        success = manager.add_rule(sensitive_term, replacement)
        
        if success:
            logger.info(f"Successfully added redaction rule")
            return True, _create_success_message(sensitive_term, replacement)
        else:
            error_msg = f"❌ ERROR: Failed to add redaction rule for term: '{sensitive_term}'"
            logger.error(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"❌ ERROR: Exception while adding redaction rule: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def redaction_tool_func(sensitive_term: str, replacement: str = "[REDACTED]") -> str:
    """
    Add a redaction rule to redacted.json for sensitive content.
    
    Args:
        sensitive_term (str): The exact sensitive string to redact
        replacement (str): Replacement text (default: [REDACTED])
        
    Returns:
        str: Success or error message about the redaction rule
    """
    # Validate dependencies
    valid, error_msg = _validate_redaction_dependencies()
    if not valid:
        return error_msg
    
    # Validate input
    valid, error_msg = _validate_sensitive_term(sensitive_term)
    if not valid:
        return error_msg
    
    # Add redaction rule safely
    success, result_msg = _add_redaction_rule_safely(sensitive_term, replacement)
    return result_msg

class RedactionTool(BaseHeavenTool):
    name = "RedactionTool"
    description = "Adds sensitive strings to the redaction rules in redacted.json. Use this when you identify sensitive information that should be redacted before publication. The sensitive_term must match exactly as it appears in the text."
    func = redaction_tool_func
    args_schema = RedactionToolArgsSchema
    is_async = False