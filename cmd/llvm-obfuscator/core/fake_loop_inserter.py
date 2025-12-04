from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class FakeLoop:
    loop_type: str
    location: str
    code_snippet: str


class FakeLoopGenerator:
    """Generates and inserts non-functional loops for code obfuscation."""

    def __init__(self, seed: int = 42) -> None:
        self._rand = random.Random(seed)
        self._var_counter = 0

    def _unique_var(self, prefix: str) -> str:
        """Generate unique variable name to avoid conflicts."""
        self._var_counter += 1
        return f"_fl_{prefix}_{self._var_counter}_{self._rand.randint(1000, 9999)}"

    def generate(self, count: int, filename: str) -> List[FakeLoop]:
        """Generate fake loop structures (for reporting purposes)."""
        loops: List[FakeLoop] = []
        for index in range(count):
            loop_type = self._rand.choice(["while", "for", "do-while"])
            values = [self._rand.randint(1, 100) for _ in range(4)]
            var = self._unique_var(f"v{index}")

            if loop_type == "while":
                modulus = max(1, values[3])
                snippet = (
                    f"{{ int {var} = {values[0]}; "
                    f"while({var} < {values[1]}) {{ "
                    f"{var} += {values[2]}; "
                    f"if(({var} % {modulus}) == 0) {{ break; }} }} }}"
                )
            elif loop_type == "for":
                snippet = (
                    f"{{ for(int {var} = {values[0]}; {var} < {values[1]}; ++{var}) {{ "
                    f"volatile int _sink_{self._var_counter} = {var} * {values[2]}; }} }}"
                )
            else:  # do-while
                snippet = (
                    f"{{ int {var} = {values[0]}; "
                    f"do {{ {var} += {values[1]}; }} "
                    f"while({var} < {values[2]}); }}"
                )

            location = f"{filename}:fake_loop_{index}"
            loops.append(FakeLoop(loop_type=loop_type, location=location, code_snippet=snippet))
        return loops

    def _find_function_bodies(self, content: str) -> List[Tuple[int, int]]:
        """
        Find positions of function bodies in C/C++ source code.
        Returns list of (start, end) positions of function body interiors.
        """
        positions = []

        # Pattern to match function definitions (simplified)
        # Looks for: return_type function_name(params) { ... }
        # We find opening braces that follow function signatures

        # First, remove string literals and comments to avoid false matches
        cleaned = self._remove_strings_and_comments(content)

        # Find potential function starts by looking for pattern:
        # word (stuff) {
        func_pattern = re.compile(
            r'\b(\w+)\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )

        for match in func_pattern.finditer(cleaned):
            func_name = match.group(1)
            # Skip if it looks like a control structure
            if func_name in ('if', 'while', 'for', 'switch', 'catch'):
                continue

            brace_start = match.end() - 1  # Position of '{'

            # Find matching closing brace
            brace_end = self._find_matching_brace(content, brace_start)
            if brace_end > brace_start:
                # Return interior of function (after { and before })
                positions.append((brace_start + 1, brace_end))

        return positions

    def _remove_strings_and_comments(self, content: str) -> str:
        """Remove string literals and comments to simplify parsing."""
        result = []
        i = 0
        n = len(content)

        while i < n:
            # Single-line comment
            if i < n - 1 and content[i:i+2] == '//':
                while i < n and content[i] != '\n':
                    result.append(' ')
                    i += 1
            # Multi-line comment
            elif i < n - 1 and content[i:i+2] == '/*':
                result.append(' ')
                result.append(' ')
                i += 2
                while i < n - 1 and content[i:i+2] != '*/':
                    result.append(' ')
                    i += 1
                if i < n - 1:
                    result.append(' ')
                    result.append(' ')
                    i += 2
            # String literal
            elif content[i] == '"':
                result.append(' ')
                i += 1
                while i < n and content[i] != '"':
                    if content[i] == '\\' and i + 1 < n:
                        result.append(' ')
                        result.append(' ')
                        i += 2
                    else:
                        result.append(' ')
                        i += 1
                if i < n:
                    result.append(' ')
                    i += 1
            # Character literal
            elif content[i] == "'":
                result.append(' ')
                i += 1
                while i < n and content[i] != "'":
                    if content[i] == '\\' and i + 1 < n:
                        result.append(' ')
                        result.append(' ')
                        i += 2
                    else:
                        result.append(' ')
                        i += 1
                if i < n:
                    result.append(' ')
                    i += 1
            else:
                result.append(content[i])
                i += 1

        return ''.join(result)

    def _find_matching_brace(self, content: str, start: int) -> int:
        """Find the matching closing brace for an opening brace at position start."""
        if start >= len(content) or content[start] != '{':
            return -1

        depth = 1
        i = start + 1
        n = len(content)

        while i < n and depth > 0:
            # Skip string literals
            if content[i] == '"':
                i += 1
                while i < n and content[i] != '"':
                    if content[i] == '\\' and i + 1 < n:
                        i += 2
                    else:
                        i += 1
                i += 1
                continue
            # Skip character literals
            if content[i] == "'":
                i += 1
                while i < n and content[i] != "'":
                    if content[i] == '\\' and i + 1 < n:
                        i += 2
                    else:
                        i += 1
                i += 1
                continue
            # Skip single-line comments
            if i < n - 1 and content[i:i+2] == '//':
                while i < n and content[i] != '\n':
                    i += 1
                continue
            # Skip multi-line comments
            if i < n - 1 and content[i:i+2] == '/*':
                i += 2
                while i < n - 1 and content[i:i+2] != '*/':
                    i += 1
                i += 2
                continue

            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
            i += 1

        return i - 1 if depth == 0 else -1

    def _find_safe_insertion_points(self, content: str, func_start: int, func_end: int) -> List[int]:
        """
        Find safe positions to insert fake loops within a function body.
        Safe positions are after semicolons or closing braces that end statements.
        """
        positions = []
        func_content = content[func_start:func_end]

        # Find positions after semicolons (end of statements)
        for i, char in enumerate(func_content):
            if char == ';':
                # Check if this is at a reasonable nesting level
                pos = func_start + i + 1
                # Skip if inside a for loop header
                before = func_content[max(0, i-50):i]
                if 'for' in before and before.count('(') > before.count(')'):
                    continue
                positions.append(pos)
            elif char == '}':
                # After closing brace of a block
                pos = func_start + i + 1
                positions.append(pos)

        return positions

    def insert_fake_loops(
        self,
        source_path: Path,
        count: int,
        output_path: Optional[Path] = None
    ) -> Tuple[str, List[FakeLoop]]:
        """
        Insert fake loops into source code file.

        Args:
            source_path: Path to source file
            count: Number of fake loops to insert
            output_path: Optional output path (if None, returns modified content)

        Returns:
            Tuple of (modified source content, list of inserted fake loops)
        """
        content = source_path.read_text(encoding='utf-8', errors='ignore')

        if count <= 0:
            return content, []

        # Generate the fake loops
        fake_loops = self.generate(count, source_path.name)

        # Find function bodies
        func_bodies = self._find_function_bodies(content)

        if not func_bodies:
            # No functions found, return original content
            return content, []

        # Collect all safe insertion points across all functions
        all_insertion_points = []
        for func_start, func_end in func_bodies:
            points = self._find_safe_insertion_points(content, func_start, func_end)
            all_insertion_points.extend(points)

        if not all_insertion_points:
            # No safe insertion points found
            return content, []

        # Randomly select insertion points (don't insert more loops than we have points)
        num_insertions = min(count, len(all_insertion_points))
        selected_points = self._rand.sample(all_insertion_points, num_insertions)

        # Sort in reverse order so we can insert without invalidating positions
        selected_points.sort(reverse=True)

        # Insert loops at selected positions
        modified_content = content
        inserted_loops = []

        for i, pos in enumerate(selected_points):
            if i < len(fake_loops):
                loop = fake_loops[i]
                # Add newlines for readability
                insert_code = f"\n    /* dead code */ {loop.code_snippet}\n"
                modified_content = (
                    modified_content[:pos] +
                    insert_code +
                    modified_content[pos:]
                )
                # Update location with actual line number
                line_num = modified_content[:pos].count('\n') + 1
                loop.location = f"{source_path.name}:{line_num}"
                inserted_loops.append(loop)

        # Write to output if specified
        if output_path:
            output_path.write_text(modified_content, encoding='utf-8')

        return modified_content, inserted_loops
