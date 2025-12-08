"""VM Obfuscation Configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class VMConfig:
    """Configuration for VM-based code obfuscation.

    Attributes:
        enabled: Whether VM obfuscation is enabled.
        functions: List of function names to virtualize. Empty means auto-detect.
        timeout: Maximum seconds before killing the VM process.
        complexity: VM complexity level (1-3 scale).
        fallback_on_error: If True, return original IR on any error.
    """
    enabled: bool = False
    functions: List[str] = field(default_factory=list)
    timeout: int = 60
    complexity: int = 1
    fallback_on_error: bool = True
