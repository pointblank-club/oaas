from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class AntiDebugCheck:
    check_type: str
    location: str
    code_snippet: str


class AntiDebugInjector:
    """Injects anti-debugging code to detect and prevent debugging."""

    def __init__(self, seed: int = 42) -> None:
        self._rand = random.Random(seed)
        self._var_counter = 0

    def _unique_var(self, prefix: str) -> str:
        """Generate unique variable name to avoid conflicts."""
        self._var_counter += 1
        return f"_ad_{prefix}_{self._var_counter}_{self._rand.randint(1000, 9999)}"

    def _generate_ptrace_check(self) -> str:
        """Generate ptrace-based debugger detection code."""
        var = self._unique_var("ptrace")
        # ptrace(PTRACE_TRACEME, 0, 0, 0) returns -1 if already being traced
        # When GDB runs a program, it attaches via ptrace, so PTRACE_TRACEME will fail
        # Use volatile to prevent compiler optimization
        # NOTE: Removed #ifdef __linux__ - we always target Linux and the ifdef
        # might not work correctly during MLIR clang compilation
        return f"""
    /* Anti-debug: ptrace check */
    {{
        // Try to trace ourselves - if we're already being traced, this fails
        volatile long {var}_result = ptrace(PTRACE_TRACEME, 0, 0, 0);
        if ({var}_result == -1) {{
            /* Debugger detected via ptrace */
            write(2, "DBG\\n", 4);  // Write to stderr before exit
            _exit(1);
        }}
    }}"""

    def _generate_proc_status_check(self) -> str:
        """Generate /proc/self/status TracerPid check."""
        var_fd = self._unique_var("fd")
        var_buf = self._unique_var("buf")
        var_line = self._unique_var("line")
        return f"""
    /* Anti-debug: /proc/self/status TracerPid check */
    {{
        FILE* {var_fd} = fopen("/proc/self/status", "r");
        if ({var_fd}) {{
            char {var_buf}[256];
            while (fgets({var_buf}, sizeof({var_buf}), {var_fd})) {{
                if (strncmp({var_buf}, "TracerPid:", 10) == 0) {{
                    int {var_line} = atoi({var_buf} + 10);
                    if ({var_line} != 0) {{
                        /* Debugger detected via TracerPid */
                        fclose({var_fd});
                        write(2, "TRC\\n", 4);  // Write to stderr before exit
                        _exit(1);
                    }}
                    break;
                }}
            }}
            fclose({var_fd});
        }}
    }}"""

    def _generate_parent_check(self) -> str:
        """Generate parent process name check."""
        var_ppid = self._unique_var("ppid")
        var_cmdline = self._unique_var("cmdline")
        var_fd = self._unique_var("fd")
        var_buf = self._unique_var("buf")
        return f"""
    /* Anti-debug: parent process check */
    {{
        pid_t {var_ppid} = getppid();
        char {var_cmdline}[256];
        snprintf({var_cmdline}, sizeof({var_cmdline}), "/proc/%d/cmdline", (int){var_ppid});
        FILE* {var_fd} = fopen({var_cmdline}, "r");
        if ({var_fd}) {{
            char {var_buf}[256];
            if (fgets({var_buf}, sizeof({var_buf}), {var_fd})) {{
                if (strstr({var_buf}, "gdb") || strstr({var_buf}, "lldb") || 
                    strstr({var_buf}, "strace") || strstr({var_buf}, "ltrace")) {{
                    /* Debugger detected in parent process */
                    fclose({var_fd});
                    _exit(1);
                }}
            }}
            fclose({var_fd});
        }}
    }}"""

    def _generate_time_check(self) -> str:
        """Generate timing-based debugger detection."""
        var_start = self._unique_var("start")
        var_end = self._unique_var("end")
        var_diff = self._unique_var("diff")
        return f"""
    /* Anti-debug: timing check */
    {{
        struct timespec {var_start}, {var_end};
        clock_gettime(CLOCK_MONOTONIC, &{var_start});
        /* Dummy operation */
        volatile int {var_diff} = 0;
        for (int i = 0; i < 1000; i++) {{ {var_diff} += i; }}
        clock_gettime(CLOCK_MONOTONIC, &{var_end});
        long {var_diff}_ns = ({var_end}.tv_sec - {var_start}.tv_sec) * 1000000000L + 
                             ({var_end}.tv_nsec - {var_start}.tv_nsec);
        /* If execution took too long, likely being debugged */
        if ({var_diff}_ns > 10000000) {{  /* 10ms threshold */
            _exit(1);
        }}
    }}"""

    def generate_anti_debug_code(self, techniques: List[str]) -> str:
        """Generate anti-debugging code using specified techniques."""
        code_parts = []
        
        if "ptrace" in techniques:
            code_parts.append(self._generate_ptrace_check())
        if "proc_status" in techniques:
            code_parts.append(self._generate_proc_status_check())
        if "parent_check" in techniques:
            code_parts.append(self._generate_parent_check())
        if "timing" in techniques:
            code_parts.append(self._generate_time_check())
        
        return "\n".join(code_parts)

    def _find_function_bodies(self, content: str) -> List[Tuple[int, int]]:
        """
        Find positions of function bodies in C/C++ source code.
        Returns list of (start, end) positions of function body interiors.
        """
        positions = []
        cleaned = self._remove_strings_and_comments(content)
        
        # Improved pattern to handle both C and C++ functions
        # Matches: return_type function_name(params) { or return_type::function_name(params) {
        func_pattern = re.compile(
            r'(?:\w+\s+)?(?:\w+::)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{',
            re.MULTILINE
        )
        
        for match in func_pattern.finditer(cleaned):
            func_name = match.group(1)
            # Skip control structures and C++ keywords
            if func_name in ('if', 'while', 'for', 'switch', 'catch', 'try', 'else'):
                continue
            
            brace_start = match.end() - 1
            brace_end = self._find_matching_brace(content, brace_start)
            if brace_end > brace_start:
                positions.append((brace_start + 1, brace_end))
        
        return positions

    def _remove_strings_and_comments(self, content: str) -> str:
        """Remove string literals and comments to simplify parsing."""
        result = []
        i = 0
        n = len(content)
        
        while i < n:
            if i < n - 1 and content[i:i+2] == '//':
                while i < n and content[i] != '\n':
                    result.append(' ')
                    i += 1
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
            if content[i] == '"':
                i += 1
                while i < n and content[i] != '"':
                    if content[i] == '\\' and i + 1 < n:
                        i += 2
                    else:
                        i += 1
                i += 1
                continue
            if content[i] == "'":
                i += 1
                while i < n and content[i] != "'":
                    if content[i] == '\\' and i + 1 < n:
                        i += 2
                    else:
                        i += 1
                i += 1
                continue
            if i < n - 1 and content[i:i+2] == '//':
                while i < n and content[i] != '\n':
                    i += 1
                continue
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
        """Find safe positions to insert anti-debugging code (preferably at function start)."""
        positions = []
        func_content = content[func_start:func_end]
        
        # Prefer inserting at the very beginning of function body (after opening brace)
        # This ensures anti-debug checks run early
        start_pos = func_start
        
        # Skip whitespace after opening brace
        i = 0
        while i < len(func_content) and func_content[i] in ' \t\n\r':
            i += 1
        
        if i < len(func_content):
            positions.append(func_start + i)
        
        # Also find positions after semicolons (for additional checks)
        for j, char in enumerate(func_content):
            if char == ';':
                pos = func_start + j + 1
                lookback = func_content[max(0, j-300):j]
                paren_depth = lookback.count('(') - lookback.count(')')
                bracket_depth = lookback.count('[') - lookback.count(']')
                
                if paren_depth == 0 and bracket_depth == 0:
                    lookahead = func_content[j+1:min(j+100, len(func_content))]
                    lookahead_stripped = lookahead.lstrip()
                    if not (lookahead_stripped and lookahead_stripped[0] in (')', ']')):
                        positions.append(pos)
        
        return positions

    def _generate_constructor_function(self, techniques: List[str]) -> str:
        """Generate a constructor function that runs before main() - more reliable for C++."""
        anti_debug_code = self.generate_anti_debug_code(techniques)
        # Indent the anti-debug code for the function body
        indented_code = '\n'.join('    ' + line if line.strip() else line 
                                  for line in anti_debug_code.split('\n'))
        # Use __attribute__((used)) to prevent optimizer from removing the function
        # Use __attribute__((noinline)) to prevent inlining which could be optimized away
        return f"""/* Anti-debugging constructor - runs before main() */
__attribute__((constructor, used, noinline))
static void __anti_debug_init(void) {{
{indented_code}
}}
"""

    def inject_anti_debug(
        self,
        source_path: Path,
        techniques: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> Tuple[str, List[AntiDebugCheck]]:
        """
        Inject anti-debugging code into source file.
        
        Args:
            source_path: Path to source file
            techniques: List of techniques to use (default: ["ptrace", "proc_status"])
            output_path: Optional output path (if None, returns modified content)
        
        Returns:
            Tuple of (modified source content, list of inserted checks)
        """
        if techniques is None:
            techniques = ["ptrace", "proc_status"]
        
        content = source_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check if required headers are already included
        # Note: <string.h> works for both C and C++ and puts strncmp in global namespace
        has_sys_ptrace = '#include <sys/ptrace.h>' in content or '#include <sys/ptrace>' in content
        has_sys_syscall = '#include <sys/syscall.h>' in content
        has_unistd = '#include <unistd.h>' in content or '#include <unistd>' in content
        has_stdio = '#include <stdio.h>' in content or '#include <stdio>' in content
        has_errno = '#include <errno.h>' in content
        # Check for string.h (C header) - this works in both C and C++
        # Note: <string> is C++ std::string, <string.h> is C string functions
        has_string = '#include <string.h>' in content
        has_time = '#include <time.h>' in content or '#include <time>' in content
        has_sys_types = '#include <sys/types.h>' in content or '#include <sys/types>' in content
        
        # Add required headers if missing
        header_insertions = []
        if "ptrace" in techniques:
            if not has_sys_ptrace:
                header_insertions.append("#include <sys/ptrace.h>\n")
            if not has_errno:
                header_insertions.append("#include <errno.h>\n")
        if not has_unistd:
            header_insertions.append("#include <unistd.h>\n")
        if ("proc_status" in techniques or "parent_check" in techniques) and not has_stdio:
            header_insertions.append("#include <stdio.h>\n")
        # Always add string.h if proc_status or parent_check is used (they need strncmp)
        needs_strncmp = "proc_status" in techniques or "parent_check" in techniques
        # Always add string.h if needed - include guards will prevent duplicates
        if needs_strncmp:
            # Always use <string.h> - works in both C and C++, puts strncmp in global namespace
            header_insertions.append("#include <string.h>\n")
        # Add stdlib.h for atoi (used in proc_status check)
        if "proc_status" in techniques:
            header_insertions.append("#include <stdlib.h>\n")
        if "timing" in techniques and not has_time:
            header_insertions.append("#include <time.h>\n")
        if "parent_check" in techniques and not has_sys_types:
            header_insertions.append("#include <sys/types.h>\n")
        
        # Add headers after existing includes or at the top
        if header_insertions:
            # Find the last #include line (match at start of line or after whitespace)
            include_pattern = re.compile(r'^#include\s*[<"].*[>"]', re.MULTILINE)
            last_include = None
            last_include_line = -1
            for match in include_pattern.finditer(content):
                last_include = match.end()
                # Get line number
                last_include_line = content[:match.start()].count('\n')
            
            if last_include is not None:
                # Insert after last include, ensuring each header is on its own line
                # Always add newline before and after headers
                headers_text = '\n' + ''.join(header_insertions)
                content = content[:last_include] + headers_text + content[last_include:]
            else:
                # Insert at the beginning, each header on its own line
                headers_text = ''.join(header_insertions) + '\n'
                content = headers_text + content
            
            # Verify headers were added (for debugging)
            if needs_strncmp and '#include <string.h>' not in content:
                # Force add it if it's still missing
                include_pattern = re.compile(r'^#include\s*[<"].*[>"]', re.MULTILINE)
                last_include = None
                for match in include_pattern.finditer(content):
                    last_include = match.end()
                if last_include is not None:
                    content = content[:last_include] + '\n#include <string.h>\n' + content[last_include:]
                else:
                    content = '#include <string.h>\n' + content
        
        # Note: PTRACE_TRACEME is defined in sys/ptrace.h, no need to define it manually
        
        # Strategy: Inject directly into main() ONLY
        # NOTE: Constructor functions (__attribute__((constructor))) are stripped by the 
        # LLVM/MLIR obfuscation pipeline (llvm.global_ctors metadata is lost during IR transforms).
        # So we MUST inject directly into main() for the anti-debug code to survive obfuscation.
        
        # Generate inline code for main()
        inline_anti_debug_code = self.generate_anti_debug_code(techniques)
        
        # Find main() and inject at the start
        func_bodies = self._find_function_bodies(content)
        
        injection_success = False
        if func_bodies:
            # Collect insertion points (prefer main() or first function)
            all_insertion_points = []
            for func_start, func_end in func_bodies:
                points = self._find_safe_insertion_points(content, func_start, func_end)
                if points:
                    all_insertion_points.append((points[0], func_start, func_end))
            
            if all_insertion_points:
                # Insert into main() if it exists, otherwise first function
                insertion_point = None
                for point, func_start, func_end in all_insertion_points:
                    func_content_before = content[max(0, func_start-200):func_start]
                    if re.search(r'\bmain\s*\(', func_content_before):
                        insertion_point = point
                        injection_success = True
                        break
                
                if insertion_point is None:
                    # No main() found, use first function as fallback
                    insertion_point = all_insertion_points[0][0]
                    injection_success = True
                
                # Insert anti-debug code at the start of the function
                insert_code = f"\n    /* Anti-debugging protection */{inline_anti_debug_code}\n"
                content = (
                    content[:insertion_point] +
                    insert_code +
                    content[insertion_point:]
                )
        
        if not injection_success:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Could not find main() or any function to inject anti-debug code")
        
        modified_content = content
        
        # Create check metadata - use last_include position if insertion_point wasn't set
        check_location = last_include if last_include is not None else 0
        checks = [
            AntiDebugCheck(
                check_type=tech,
                location=f"{source_path.name}:constructor",
                code_snippet=tech
            )
            for tech in techniques
        ]
        
        # Write to output if specified
        if output_path:
            output_path.write_text(modified_content, encoding='utf-8')
            # Log what was written for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Anti-debug: wrote {len(modified_content)} bytes to {output_path}")
            if 'ptrace' in modified_content:
                logger.info("Anti-debug: ptrace code confirmed in output")
            if '__anti_debug_init' in modified_content:
                logger.info("Anti-debug: constructor function confirmed in output")
        
        return modified_content, checks

