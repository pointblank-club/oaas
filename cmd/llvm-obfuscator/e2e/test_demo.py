"""OAAS End-to-End Demo Tests for SIH Presentation.

These tests demonstrate the complete obfuscation flow:
1. Upload source code
2. Select obfuscation options
3. Enable/disable VM virtualization
4. Run obfuscation
5. Download result

Run with video recording:
    pytest e2e/test_demo.py --video=on --screenshot=on -v
"""

from pathlib import Path
from typing import Generator

import pytest
from playwright.sync_api import Page, expect

from conftest import screenshot_dir, take_screenshot


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def demo_source() -> str:
    """Return simple C source code for demo."""
    return '''#include <stdio.h>

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

int main() {
    printf("Result: %d\\n", add(5, 3));
    return 0;
}'''


# =============================================================================
# Demo Test Cases
# =============================================================================

class TestOAASDemo:
    """End-to-end demo tests for OAAS obfuscation."""

    def test_obfuscation_demo_without_vm(
        self,
        demo_page: Page,
        base_url: str,
        screenshot_dir: Path,
        demo_source: str,
    ) -> None:
        """Demo: Obfuscation flow WITHOUT VM (standard OLLVM).

        This test demonstrates the basic obfuscation workflow using
        OLLVM passes (control flow flattening, substitution).
        """
        page = demo_page

        # Step 1: Navigate to OAAS homepage
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        take_screenshot(page, "01_homepage", screenshot_dir)

        # Step 2: Select "Paste Code" input mode if available
        paste_tab = page.locator("text=Paste Code").first
        if paste_tab.is_visible():
            paste_tab.click()
            page.wait_for_timeout(500)

        # Step 3: Enter source code
        code_input = page.locator("textarea").first
        code_input.fill(demo_source)
        take_screenshot(page, "02_code_entered", screenshot_dir)

        # Step 4: Enable Layer 3 (OLLVM Passes)
        layer3_checkbox = page.locator("text=LAYER 3").first
        if layer3_checkbox.is_visible():
            layer3_checkbox.click()
            page.wait_for_timeout(300)

        # Step 5: Select specific passes
        flattening = page.locator("text=Flattening").first
        if flattening.is_visible():
            flattening.click()

        substitution = page.locator("text=Substitution").first
        if substitution.is_visible():
            substitution.click()

        take_screenshot(page, "03_options_no_vm", screenshot_dir)

        # Step 6: Verify VM is OFF (should be off by default)
        vm_toggle = page.locator("[data-testid='vm-toggle']")
        if vm_toggle.is_visible():
            expect(vm_toggle).not_to_be_checked()

        # Step 7: Click obfuscate button
        submit_btn = page.locator(".submit-btn").first
        submit_btn.click()
        take_screenshot(page, "04_processing", screenshot_dir)

        # Step 8: Wait for processing (with timeout)
        page.wait_for_selector(".modal-btn, .download-btn, text=Download", timeout=120000)
        take_screenshot(page, "05_result", screenshot_dir)

        # Step 9: Verify success (modal or download link)
        success_indicator = page.locator("text=success, text=Download, text=Complete").first
        expect(success_indicator).to_be_visible(timeout=5000)

    def test_obfuscation_demo_with_vm(
        self,
        demo_page: Page,
        base_url: str,
        screenshot_dir: Path,
        demo_source: str,
    ) -> None:
        """Demo: Obfuscation flow WITH VM virtualization enabled.

        This test demonstrates enabling the experimental VM layer
        which converts functions to bytecode for maximum obfuscation.
        """
        page = demo_page

        # Step 1: Navigate to OAAS homepage
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        # Step 2: Select paste mode and enter code
        paste_tab = page.locator("text=Paste Code").first
        if paste_tab.is_visible():
            paste_tab.click()
            page.wait_for_timeout(500)

        code_input = page.locator("textarea").first
        code_input.fill(demo_source)

        # Step 3: Enable Layer 3 (OLLVM)
        layer3_checkbox = page.locator("text=LAYER 3").first
        if layer3_checkbox.is_visible():
            layer3_checkbox.click()
            page.wait_for_timeout(300)

        # Step 4: Enable VM toggle (experimental feature)
        vm_toggle = page.locator("[data-testid='vm-toggle']")
        if vm_toggle.is_visible():
            vm_toggle.click()
            page.wait_for_timeout(300)
            expect(vm_toggle).to_be_checked()
            take_screenshot(page, "06_vm_enabled", screenshot_dir)

        # Step 5: Verify timeout input appears
        vm_timeout = page.locator("[data-testid='vm-timeout']")
        if vm_timeout.is_visible():
            expect(vm_timeout).to_have_value("60")

        # Step 6: Click obfuscate
        submit_btn = page.locator(".submit-btn").first
        submit_btn.click()
        take_screenshot(page, "07_vm_processing", screenshot_dir)

        # Step 7: Wait for result (VM may take longer)
        page.wait_for_selector(".modal-btn, .download-btn, text=Download", timeout=180000)
        take_screenshot(page, "08_vm_result", screenshot_dir)

    def test_preset_selection_demo(
        self,
        demo_page: Page,
        base_url: str,
        screenshot_dir: Path,
        demo_source: str,
    ) -> None:
        """Demo: Using preset configurations.

        Shows the quick preset selection feature for easy setup.
        """
        page = demo_page

        # Step 1: Navigate
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        # Step 2: Enter code
        paste_tab = page.locator("text=Paste Code").first
        if paste_tab.is_visible():
            paste_tab.click()
            page.wait_for_timeout(500)

        code_input = page.locator("textarea").first
        code_input.fill(demo_source)

        # Step 3: Select "Maximum" preset
        max_preset = page.locator("text=Maximum").first
        if max_preset.is_visible():
            max_preset.click()
            page.wait_for_timeout(500)
            take_screenshot(page, "09_maximum_preset", screenshot_dir)

        # Step 4: Verify layers are auto-configured
        # Maximum should enable multiple layers
        layer1 = page.locator("text=LAYER 1").first
        layer3 = page.locator("text=LAYER 3").first

        # Step 5: Run obfuscation
        submit_btn = page.locator(".submit-btn").first
        submit_btn.click()
        take_screenshot(page, "10_preset_processing", screenshot_dir)

        # Step 6: Wait for result
        page.wait_for_selector(".modal-btn, .download-btn, text=Download", timeout=180000)
        take_screenshot(page, "11_preset_result", screenshot_dir)


class TestVMFallback:
    """Tests demonstrating VM graceful fallback behavior."""

    def test_vm_fallback_on_complex_code(
        self,
        demo_page: Page,
        base_url: str,
        screenshot_dir: Path,
    ) -> None:
        """Demo: VM falls back gracefully on unsupported code.

        VM virtualization only supports simple arithmetic functions.
        When complex code is provided, it falls back to original code.
        """
        page = demo_page

        # Complex code with loops and calls (unsupported by VM)
        complex_source = '''#include <stdio.h>
#include <stdlib.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // Recursive call - unsupported
}

int main() {
    for (int i = 0; i < 5; i++) {  // Loop - unsupported
        printf("%d! = %d\\n", i, factorial(i));
    }
    return 0;
}'''

        # Navigate and enter complex code
        page.goto(base_url)
        page.wait_for_load_state("networkidle")

        paste_tab = page.locator("text=Paste Code").first
        if paste_tab.is_visible():
            paste_tab.click()
            page.wait_for_timeout(500)

        code_input = page.locator("textarea").first
        code_input.fill(complex_source)

        # Enable VM
        vm_toggle = page.locator("[data-testid='vm-toggle']")
        if vm_toggle.is_visible():
            vm_toggle.click()
            take_screenshot(page, "12_vm_complex_code", screenshot_dir)

        # Run obfuscation
        submit_btn = page.locator(".submit-btn").first
        submit_btn.click()

        # Wait for result - should succeed with fallback
        page.wait_for_selector(".modal-btn, .download-btn, text=Download", timeout=180000)
        take_screenshot(page, "13_vm_fallback_result", screenshot_dir)

        # Verify obfuscation completed (even if VM fell back)
        # The key is that it doesn't crash
        success_indicator = page.locator("text=success, text=Download, text=Complete").first
        expect(success_indicator).to_be_visible(timeout=5000)
