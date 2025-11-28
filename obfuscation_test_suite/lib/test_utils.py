#!/usr/bin/env python3
"""Utility functions for test suite"""

import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_command(cmd: str, timeout: int = 30) -> str:
    """Execute shell command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {cmd}")
        return ""
    except Exception as e:
        logger.warning(f"Command failed: {cmd} - {e}")
        return ""


def safe_run(cmd: str, timeout: int = 30) -> bool:
    """Execute command and return success status"""
    try:
        subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            timeout=timeout
        )
        return True
    except Exception as e:
        logger.warning(f"Command failed: {cmd} - {e}")
        return False


def file_hash(filepath: str, algorithm: str = 'sha256') -> str:
    """Calculate file hash"""
    hash_func = hashlib.new(algorithm)

    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.warning(f"Could not hash file: {filepath} - {e}")
        return ""


def extract_strings(filepath: str, min_length: int = 4) -> list:
    """Extract printable strings from binary"""
    try:
        result = run_command(f"strings -n {min_length} {filepath}")
        return [s for s in result.strip().split('\n') if s]
    except Exception as e:
        logger.warning(f"Could not extract strings: {e}")
        return []


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    try:
        return Path(filepath).stat().st_size
    except Exception as e:
        logger.warning(f"Could not get file size: {e}")
        return 0


def is_executable(filepath: str) -> bool:
    """Check if file is executable"""
    try:
        import os
        return os.access(filepath, os.X_OK)
    except:
        return False


def get_arch(filepath: str) -> str:
    """Get binary architecture"""
    try:
        result = run_command(f"file {filepath}")
        if 'x86-64' in result or 'x86_64' in result:
            return 'x86_64'
        elif 'Intel' in result or '32-bit' in result:
            return 'i386'
        elif 'ARM' in result:
            return 'ARM'
        else:
            return 'unknown'
    except:
        return 'unknown'


def get_sections(filepath: str) -> dict:
    """Get binary section information"""
    sections = {}
    try:
        result = run_command(f"objdump -h {filepath}")
        for line in result.split('\n')[4:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                sections[parts[1]] = {
                    'size': parts[2],
                    'vaddr': parts[3]
                }
    except Exception as e:
        logger.warning(f"Could not extract sections: {e}")

    return sections


def compare_outputs(out1: str, out2: str) -> bool:
    """Compare command outputs"""
    return out1.strip() == out2.strip()
