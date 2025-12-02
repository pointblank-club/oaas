"""Multi-file project support and GitHub integration for LLVM Obfuscator."""

from __future__ import annotations

import io
import logging
import re
import secrets
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import HTTPException, Request
from github import Github
from pydantic import BaseModel

from core.utils import create_logger, ensure_directory

logger = create_logger("multifile", logging.INFO)

# In-memory storage for ephemeral repo sessions
# Maps session_id -> {"repo_path": Path, "timestamp": float, "repo_name": str, "branch": str}
repo_sessions: Dict[str, Dict] = {}


class SourceFile(BaseModel):
    """Represents a single source file with its path and content."""
    path: str
    content: str


class GitHubRepoRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    access_token: Optional[str] = None


class RepoFile(BaseModel):
    path: str
    content: str
    is_binary: bool = False


class RepoData(BaseModel):
    files: List[RepoFile]
    total_files: int
    repo_name: str
    branch: str


def find_main_file(source_files: List[SourceFile]) -> Optional[SourceFile]:
    """Find the file containing the main() function."""
    main_pattern = re.compile(r'\bint\s+main\s*\(|\bvoid\s+main\s*\(')

    for file in source_files:
        # Only check C/C++ files
        ext = file.path.lower().split('.')[-1] if '.' in file.path else ''
        if ext in ['c', 'cpp', 'cc', 'cxx', 'c++']:
            if main_pattern.search(file.content):
                return file
    return None


def create_multi_file_project(source_files: List[SourceFile], project_dir: Path) -> Tuple[Path, List[Path]]:
    """Create a multi-file project structure and return the main file path and all source paths."""
    ensure_directory(project_dir)

    source_paths = []
    main_file_path = None

    # Filter only C/C++ source and header files
    valid_extensions = ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++']

    for file in source_files:
        ext = file.path.lower().split('.')[-1] if '.' in file.path else ''
        if ext not in valid_extensions:
            continue

        # Create subdirectories if needed
        file_full_path = (project_dir / file.path).resolve()

        # Security check: ensure file path is within project_dir
        try:
            file_full_path.relative_to(project_dir)
        except ValueError:
            logger.warning(f"Skipping file outside project directory: {file.path}")
            continue

        ensure_directory(file_full_path.parent)

        # Write file content
        try:
            file_full_path.write_text(file.content, encoding='utf-8')

            # Track source files (not headers)
            if ext in ['c', 'cpp', 'cc', 'cxx', 'c++']:
                source_paths.append(file_full_path)

                # Check if this is the main file
                if re.search(r'\bint\s+main\s*\(|\bvoid\s+main\s*\(', file.content):
                    main_file_path = file_full_path

        except Exception as e:
            logger.warning(f"Failed to write file {file.path}: {e}")
            continue

    if not main_file_path and source_paths:
        # If no main function found, use the first source file
        main_file_path = source_paths[0]

    return main_file_path or (project_dir / "main.c"), source_paths


def extract_repo_files(repo_url: str, branch: str = "main", access_token: Optional[str] = None) -> RepoData:
    """Extract files from a GitHub repository using zip download.
    
    This legacy endpoint downloads the repo as a zip, extracts it to a temporary location,
    reads the files, and returns them to the frontend.
    """
    temp_dir = None
    try:
        # Parse GitHub URL to get owner/repo
        if "github.com" not in repo_url:
            raise ValueError("Only GitHub repositories are supported")

        # Extract owner/repo from URL
        parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid GitHub repository URL")

        owner, repo_name = parts[0], parts[1]

        # Create temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix=f"oaas_extract_{owner}_{repo_name}_"))
        logger.info(f"Created temporary directory for extraction: {temp_dir}")
        
        # Download repository as zip archive
        zip_url = f"https://github.com/{owner}/{repo_name}/zipball/{branch}"
        logger.info(f"Downloading repository zip from: {zip_url}")
        
        # Set up headers with authentication if token provided
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        
        # Download the zip file
        response = requests.get(zip_url, headers=headers, stream=True, timeout=300)
        
        if response.status_code == 404:
            raise ValueError(f"Repository or branch not found: {owner}/{repo_name} (branch: {branch})")
        elif response.status_code == 401:
            raise ValueError("Authentication failed. Invalid access token.")
        elif response.status_code != 200:
            raise ValueError(f"Failed to download repository: HTTP {response.status_code}")
        
        # Extract the zip file
        logger.info("Extracting repository archive...")
        zip_data = io.BytesIO(response.content)
        
        with zipfile.ZipFile(zip_data) as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the extracted directory
        extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
        if not extracted_dirs:
            raise ValueError("No directory found in extracted zip")
        
        repo_path = extracted_dirs[0]
        
        # Read all relevant files from the extracted directory
        files = []
        valid_extensions = ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++', 
                          'txt', 'md', 'py', 'js', 'ts', 'json', 'yml', 'yaml', 'xml', 'html', 'css']
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                try:
                    # Get file extension
                    file_ext = file_path.suffix.lstrip('.').lower() if file_path.suffix else ''
                    
                    # Only include relevant file types
                    if file_ext in valid_extensions:
                        # Skip files larger than 1MB
                        if file_path.stat().st_size < 1024 * 1024:
                            # Get relative path from repo root
                            relative_path = file_path.relative_to(repo_path)
                            
                            # Read file content
                            file_content = file_path.read_text(encoding='utf-8', errors='ignore')
                            
                            files.append(RepoFile(
                                path=str(relative_path),
                                content=file_content,
                                is_binary=False
                            ))
                except Exception as e:
                    logger.warning(f"Skipping file {file_path}: {e}")
                    continue
        
        logger.info(f"Extracted {len(files)} files from repository")
        
        return RepoData(
            files=files,
            total_files=len(files),
            repo_name=f"{owner}/{repo_name}",
            branch=branch
        )

    except Exception as e:
        logger.error(f"Failed to extract repository files: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to access repository: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary extraction directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")


def get_repo_branches(repo_url: str, access_token: Optional[str] = None) -> List[str]:
    """Get list of branches for a GitHub repository."""
    try:
        # Parse GitHub URL to get owner/repo
        if "github.com" not in repo_url:
            raise ValueError("Only GitHub repositories are supported")

        parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid GitHub repository URL")

        owner, repo_name = parts[0], parts[1]

        # Initialize GitHub client
        if access_token:
            g = Github(access_token)
        else:
            g = Github()  # Anonymous access for public repos

        repo = g.get_repo(f"{owner}/{repo_name}")
        branches = [branch.name for branch in repo.get_branches()]

        return branches

    except Exception as e:
        logger.error(f"Failed to get repository branches: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to access repository: {str(e)}")


def get_session_token(request: Request, session_cookie_name: str, user_sessions: dict, token_ttl: int) -> Optional[str]:
    """Extract and validate session token from cookie."""
    import time
    session_id = request.cookies.get(session_cookie_name)
    if not session_id or session_id not in user_sessions:
        return None

    session = user_sessions[session_id]
    # Check if token expired
    if time.time() - session["timestamp"] > token_ttl:
        del user_sessions[session_id]
        return None

    return session["token"]


def clone_repo_to_temp(repo_url: str, branch: str = "main", access_token: Optional[str] = None) -> Tuple[str, Path]:
    """Clone GitHub repository to temporary directory using zip download and return session ID and path.
    
    This method downloads the repository as a zip archive and extracts it, which is much faster
    than downloading files one-by-one via the API for large repositories.
    
    Args:
        repo_url: GitHub repository URL
        branch: Branch to clone (default: main)
        access_token: Optional GitHub access token for private repos
        
    Returns:
        Tuple of (session_id, repo_path)
    """
    try:
        # Parse GitHub URL
        if "github.com" not in repo_url:
            raise ValueError("Only GitHub repositories are supported")
        
        parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
        
        owner, repo_name = parts[0], parts[1]
        
        # Create temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix=f"oaas_repo_{owner}_{repo_name}_"))
        logger.info(f"Created temporary directory for repo: {temp_dir}")
        
        # Download repository as zip archive
        # GitHub provides zipball endpoint: https://github.com/{owner}/{repo}/zipball/{ref}
        zip_url = f"https://github.com/{owner}/{repo_name}/zipball/{branch}"
        logger.info(f"Downloading repository zip from: {zip_url}")
        
        # Set up headers with authentication if token provided
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        
        # Download the zip file
        response = requests.get(zip_url, headers=headers, stream=True, timeout=300)
        
        if response.status_code == 404:
            raise ValueError(f"Repository or branch not found: {owner}/{repo_name} (branch: {branch})")
        elif response.status_code == 401:
            raise ValueError("Authentication failed. Invalid access token.")
        elif response.status_code != 200:
            raise ValueError(f"Failed to download repository: HTTP {response.status_code}")
        
        # Extract the zip file
        logger.info("Extracting repository archive...")
        zip_data = io.BytesIO(response.content)
        
        with zipfile.ZipFile(zip_data) as zip_ref:
            # GitHub zipballs have a top-level directory like "owner-repo-commit_sha"
            # We need to extract it and use that directory
            zip_ref.extractall(temp_dir)
        
        # Find the extracted directory (should be only one top-level directory)
        extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
        
        if not extracted_dirs:
            raise ValueError("No directory found in extracted zip")
        
        # Use the first (and should be only) directory as repo_path
        repo_path = extracted_dirs[0]
        logger.info(f"Extracted to: {repo_path}")
        
        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store session
        repo_sessions[session_id] = {
            "repo_path": repo_path,
            "timestamp": time.time(),
            "repo_name": f"{owner}/{repo_name}",
            "branch": branch,
        }
        
        logger.info(f"Repository downloaded successfully: {owner}/{repo_name} (branch: {branch})")
        logger.info(f"Session ID: {session_id}, Files stored in: {repo_path}")
        
        return session_id, repo_path
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp directory if created
        if 'temp_dir' in locals() and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")
        
        logger.error(f"Failed to clone repository: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")


def get_repo_session(session_id: str) -> Optional[Dict]:
    """Get repository session by ID."""
    return repo_sessions.get(session_id)


def cleanup_repo_session(session_id: str) -> bool:
    """Clean up repository session and delete temporary files.
    
    This function cleans up:
    1. The repo_path directory (e.g., /tmp/oaas_repo_xxx/owner-repo-sha/)
    2. The parent temp directory if it's now empty (e.g., /tmp/oaas_repo_xxx/)
    
    Args:
        session_id: Repository session ID
        
    Returns:
        True if cleanup successful, False otherwise
    """
    if session_id not in repo_sessions:
        logger.warning(f"Session not found: {session_id}")
        return False
    
    session = repo_sessions[session_id]
    repo_path = session["repo_path"]
    
    try:
        # Get parent temp directory before deleting repo_path
        parent_temp_dir = repo_path.parent if repo_path else None
        
        # Delete the repository directory
        if repo_path and repo_path.exists():
            shutil.rmtree(repo_path)
            logger.info(f"Deleted temporary repository: {repo_path}")
        
        # Also clean up the parent temp directory if it's empty
        # The repo_path is something like /tmp/oaas_repo_xxx/owner-repo-sha/
        # We want to also delete /tmp/oaas_repo_xxx/ if it's now empty
        if parent_temp_dir and parent_temp_dir.exists():
            if parent_temp_dir.name.startswith("oaas_repo_") or parent_temp_dir.name.startswith("oaas_extract_"):
                remaining = list(parent_temp_dir.iterdir())
                if not remaining:
                    shutil.rmtree(parent_temp_dir)
                    logger.info(f"Deleted parent temp directory: {parent_temp_dir}")
        
        # Remove from sessions
        del repo_sessions[session_id]
        logger.info(f"Cleaned up session: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cleanup session {session_id}: {e}")
        return False


def cleanup_old_sessions(max_age_seconds: int = 3600) -> int:
    """Clean up old repository sessions.
    
    Args:
        max_age_seconds: Maximum age in seconds (default: 1 hour)
        
    Returns:
        Number of sessions cleaned up
    """
    current_time = time.time()
    sessions_to_cleanup = []
    
    for session_id, session in repo_sessions.items():
        if current_time - session["timestamp"] > max_age_seconds:
            sessions_to_cleanup.append(session_id)
    
    cleaned = 0
    for session_id in sessions_to_cleanup:
        if cleanup_repo_session(session_id):
            cleaned += 1
    
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} old repository sessions")
    
    return cleaned
