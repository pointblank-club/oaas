"""Playwright fixtures for OAAS E2E demo tests.

Configures video recording and screenshots for SIH presentation.
"""

import os
from pathlib import Path

import pytest
from playwright.sync_api import Browser, BrowserContext, Page


# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.environ.get("OAAS_URL", "https://oaas.pointblank.club")
ARTIFACTS_DIR = Path(__file__).parent / "demo_artifacts"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def base_url() -> str:
    """Get the base URL for OAAS (configurable via OAAS_URL env var)."""
    return BASE_URL


@pytest.fixture(scope="session")
def screenshot_dir() -> Path:
    """Create and return the screenshot directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


@pytest.fixture(scope="function")
def demo_context(browser: Browser, screenshot_dir: Path) -> BrowserContext:
    """Create browser context with video recording enabled.

    Video is saved to demo_artifacts/ for presentation use.
    """
    context = browser.new_context(
        record_video_dir=str(screenshot_dir),
        record_video_size={"width": 1280, "height": 720},
        viewport={"width": 1280, "height": 720},
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def demo_page(demo_context: BrowserContext) -> Page:
    """Create a new page for demo tests."""
    page = demo_context.new_page()
    yield page
    page.close()


@pytest.fixture(scope="session")
def test_source_file() -> Path:
    """Path to the demo input C file."""
    return Path(__file__).parent / "test_sources" / "demo_input.c"


# =============================================================================
# Helper Functions
# =============================================================================

def take_screenshot(page: Page, name: str, screenshot_dir: Path) -> None:
    """Take a screenshot with a descriptive name."""
    filepath = screenshot_dir / f"{name}.png"
    page.screenshot(path=str(filepath), full_page=False)
    print(f"  Screenshot: {filepath}")
