"""
HEAVEN Acolyte - Writer of Chain Gospel (Heavenly Scripture)

The acolyte is a faithful agent that writes perfect HEAVEN scripts (chain gospel)
for any domain and task. It knows all HEAVEN patterns, best practices, and 
domain-specific requirements.

Usage:
    from heaven_base.acolyte import write_chain_gospel
    
    await write_chain_gospel(
        domain="file_editing",
        task="implement specific TODO in Python file",
        output_dir="~/chain_gospel/"
    )
"""

from .gospel_writer import write_chain_gospel, list_domains, preview_gospel

__all__ = ["write_chain_gospel", "list_domains", "preview_gospel"]