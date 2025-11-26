"""
Indirect Call Obfuscator

Transforms direct function calls into indirect calls through function pointers
to hide which functions are being called, making reverse engineering harder.

Supports:
- Standard library functions (printf, strcmp, malloc, etc.)
- Custom user-defined functions
- Both C and C++ code
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class IndirectCallObfuscator:
    """Obfuscates function calls by converting them to indirect calls via function pointers."""

    # Common standard library functions to obfuscate
    STDLIB_FUNCTIONS = {
        # stdio.h
        'printf', 'fprintf', 'sprintf', 'snprintf', 'scanf', 'fscanf', 'sscanf',
        'puts', 'fputs', 'gets', 'fgets', 'putchar', 'getchar',
        'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell', 'rewind',
        'feof', 'ferror', 'fflush',

        # string.h
        'strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy', 'strcat', 'strncat',
        'strchr', 'strrchr', 'strstr', 'strtok', 'memcpy', 'memmove', 'memset',
        'memcmp', 'memchr',

        # stdlib.h
        'malloc', 'calloc', 'realloc', 'free', 'atoi', 'atol', 'atof',
        'strtol', 'strtoul', 'strtod', 'rand', 'srand', 'exit', 'abort',
        'system', 'getenv',

        # math.h
        'sin', 'cos', 'tan', 'sqrt', 'pow', 'exp', 'log', 'log10',
        'ceil', 'floor', 'fabs', 'fmod',

        # time.h
        'time', 'clock', 'difftime', 'mktime', 'localtime', 'gmtime',
        'asctime', 'ctime', 'strftime',

        # unistd.h
        'read', 'write', 'close', 'lseek', 'access', 'unlink',
        'chdir', 'getcwd', 'getpid', 'sleep',
    }

    def __init__(self, obfuscate_stdlib: bool = True, obfuscate_custom: bool = True):
        """
        Initialize the indirect call obfuscator.

        Args:
            obfuscate_stdlib: Whether to obfuscate standard library functions
            obfuscate_custom: Whether to obfuscate custom user-defined functions
        """
        self.obfuscate_stdlib = obfuscate_stdlib
        self.obfuscate_custom = obfuscate_custom
        self.function_pointers: Dict[str, str] = {}  # function_name -> pointer_name
        self.custom_functions: Set[str] = set()

    def obfuscate(self, source_code: str, source_file: Path) -> Tuple[str, Dict]:
        """
        Obfuscate function calls in the source code.

        Args:
            source_code: The C/C++ source code
            source_file: Path to the source file

        Returns:
            Tuple of (transformed_source, metadata)
        """
        logger.info("Starting indirect call obfuscation")

        # Step 1: Detect custom functions (if enabled)
        if self.obfuscate_custom:
            self.custom_functions = self._detect_custom_functions(source_code)
            logger.info(f"Detected {len(self.custom_functions)} custom functions")

        # Step 2: Determine which functions to obfuscate
        functions_to_obfuscate = set()
        if self.obfuscate_stdlib:
            functions_to_obfuscate.update(self.STDLIB_FUNCTIONS)
        if self.obfuscate_custom:
            functions_to_obfuscate.update(self.custom_functions)

        # Step 3: Find actually used functions in the code
        used_functions = self._find_used_functions(source_code, functions_to_obfuscate)
        logger.info(f"Found {len(used_functions)} functions to obfuscate: {used_functions}")

        # Step 4: Generate function pointer declarations
        pointer_declarations = self._generate_pointer_declarations(used_functions, source_code)

        # Step 5: Generate initialization code
        init_code = self._generate_initialization_code(used_functions)

        # Step 6: Transform function calls to indirect calls
        transformed_code = self._transform_calls(source_code, used_functions)

        # Step 7: Inject declarations and initialization
        final_code = self._inject_obfuscation_code(
            transformed_code,
            pointer_declarations,
            init_code
        )

        metadata = {
            'obfuscated_stdlib_functions': len([f for f in used_functions if f in self.STDLIB_FUNCTIONS]),
            'obfuscated_custom_functions': len([f for f in used_functions if f in self.custom_functions]),
            'total_obfuscated': len(used_functions),
            'function_pointers': self.function_pointers,
        }

        logger.info(f"Indirect call obfuscation complete: {metadata}")
        return final_code, metadata

    def _detect_custom_functions(self, source_code: str) -> Set[str]:
        """Detect custom function definitions in the source code."""
        functions = set()

        # Match function definitions: return_type function_name(params) {
        # This is a simplified regex - may need enhancement for complex cases
        pattern = r'\b(?:static\s+)?(?:inline\s+)?(?:const\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*\{'

        for match in re.finditer(pattern, source_code):
            return_type = match.group(1).strip()
            func_name = match.group(2).strip()

            # Skip main function and common keywords
            if func_name != 'main' and func_name not in ['if', 'for', 'while', 'switch']:
                # Skip if it looks like a type name (starts with capital or contains struct/union)
                if not return_type.startswith(('struct', 'union', 'enum')) and func_name[0].islower():
                    functions.add(func_name)

        return functions

    def _find_used_functions(self, source_code: str, candidate_functions: Set[str]) -> Set[str]:
        """Find which candidate functions are actually called in the code."""
        used = set()

        for func_name in candidate_functions:
            # Look for function calls: function_name(
            pattern = rf'\b{re.escape(func_name)}\s*\('
            if re.search(pattern, source_code):
                used.add(func_name)

        return used

    def _generate_pointer_declarations(self, functions: Set[str], source_code: str) -> str:
        """Generate function pointer type declarations with forward declarations for custom functions."""
        declarations = []
        declarations.append("/* Indirect Call Obfuscation - Function Pointers */")

        # Add forward declarations for custom functions
        custom_funcs = [f for f in functions if f in self.custom_functions]
        if custom_funcs:
            declarations.append("\n/* Forward declarations for custom functions */")
            for func_name in sorted(custom_funcs):
                signature_full = self._infer_function_signature(func_name, source_code)
                if signature_full:
                    # Extract just the function signature without the pointer syntax
                    # Pattern: return_type (*ptr_name)(params) -> return_type func_name(params)
                    match = re.match(r'(.+?)\s*\(\*__fptr_\w+\)\((.+)\)', signature_full)
                    if match:
                        return_type = match.group(1).strip()
                        params = match.group(2).strip()
                        declarations.append(f"{return_type} {func_name}({params});")

        declarations.append("\n/* Function pointer declarations */")

        for func_name in sorted(functions):
            # Generate a unique pointer name
            ptr_name = f"__fptr_{func_name}"
            self.function_pointers[func_name] = ptr_name

            # Try to infer function signature (simplified - may need enhancement)
            signature = self._infer_function_signature(func_name, source_code)

            if signature:
                declarations.append(f"static {signature} = NULL;")
            else:
                # Fallback: use void* for unknown signatures
                declarations.append(f"static void* {ptr_name} = NULL;")

        return "\n".join(declarations)

    def _infer_function_signature(self, func_name: str, source_code: str) -> str:
        """
        Try to infer the function signature from the source code.
        This is a simplified implementation - production code would need proper C parsing.
        """
        # For standard library functions, we can hardcode known signatures
        stdlib_signatures = {
            'printf': 'int (*__fptr_printf)(const char*, ...)',
            'fprintf': 'int (*__fptr_fprintf)(FILE*, const char*, ...)',
            'sprintf': 'int (*__fptr_sprintf)(char*, const char*, ...)',
            'scanf': 'int (*__fptr_scanf)(const char*, ...)',
            'strlen': 'size_t (*__fptr_strlen)(const char*)',
            'strcmp': 'int (*__fptr_strcmp)(const char*, const char*)',
            'strcpy': 'char* (*__fptr_strcpy)(char*, const char*)',
            'strcat': 'char* (*__fptr_strcat)(char*, const char*)',
            'memcpy': 'void* (*__fptr_memcpy)(void*, const void*, size_t)',
            'memset': 'void* (*__fptr_memset)(void*, int, size_t)',
            'malloc': 'void* (*__fptr_malloc)(size_t)',
            'free': 'void (*__fptr_free)(void*)',
            'exit': 'void (*__fptr_exit)(int)',
        }

        if func_name in stdlib_signatures:
            return stdlib_signatures[func_name]

        # For custom functions, try to extract from definition
        pattern = rf'(\w+(?:\s*\*)?)\s+{re.escape(func_name)}\s*\(([^)]*)\)'
        match = re.search(pattern, source_code)

        if match:
            return_type = match.group(1).strip()
            params = match.group(2).strip()
            ptr_name = f"__fptr_{func_name}"
            return f"{return_type} (*{ptr_name})({params})"

        return None

    def _generate_initialization_code(self, functions: Set[str]) -> str:
        """Generate initialization code for function pointers."""
        init_lines = []
        init_lines.append("/* Initialize function pointers */")
        init_lines.append("__attribute__((constructor)) static void __init_function_pointers(void) {")

        for func_name in sorted(functions):
            ptr_name = self.function_pointers[func_name]
            init_lines.append(f"    {ptr_name} = (void*)&{func_name};")

        init_lines.append("}")

        return "\n".join(init_lines)

    def _transform_calls(self, source_code: str, functions: Set[str]) -> str:
        """
        Transform direct function calls to indirect calls.

        IMPORTANT: Only replace function CALLS, not function DEFINITIONS.
        """
        transformed = source_code

        for func_name in functions:
            ptr_name = self.function_pointers[func_name]

            # Split by lines to avoid replacing function definitions
            lines = transformed.split('\n')
            new_lines = []

            for line in lines:
                # Check if this line is a function definition
                # Pattern: return_type function_name(params) { or return_type function_name(params);
                definition_pattern = rf'\b(?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:\w+(?:\s*\*)?)\s+{re.escape(func_name)}\s*\([^)]*\)\s*(?:\{{|;)'

                if re.search(definition_pattern, line):
                    # This is a function definition - don't replace
                    new_lines.append(line)
                else:
                    # This is potentially a function call - replace it
                    # Use word boundary to avoid partial matches
                    pattern = rf'\b{re.escape(func_name)}\s*\('
                    replacement = f'{ptr_name}('
                    new_line = re.sub(pattern, replacement, line)
                    new_lines.append(new_line)

            transformed = '\n'.join(new_lines)

        return transformed

    def _inject_obfuscation_code(self, source_code: str, declarations: str, init_code: str) -> str:
        """Inject function pointer declarations and initialization into the source code."""
        # Find a good injection point (after #includes, before first function)
        lines = source_code.split('\n')

        # Find last #include or #define
        last_include_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include') or line.strip().startswith('#define'):
                last_include_idx = i

        # Inject after includes
        injection_point = last_include_idx + 1

        injection = [
            "",
            declarations,
            "",
            init_code,
            ""
        ]

        result_lines = lines[:injection_point] + injection + lines[injection_point:]
        return '\n'.join(result_lines)


def obfuscate_indirect_calls(
    source_code: str,
    source_file: Path,
    obfuscate_stdlib: bool = True,
    obfuscate_custom: bool = True
) -> Tuple[str, Dict]:
    """
    Convenience function to obfuscate function calls.

    Args:
        source_code: The C/C++ source code
        source_file: Path to the source file
        obfuscate_stdlib: Whether to obfuscate standard library functions
        obfuscate_custom: Whether to obfuscate custom functions

    Returns:
        Tuple of (transformed_source, metadata)
    """
    obfuscator = IndirectCallObfuscator(
        obfuscate_stdlib=obfuscate_stdlib,
        obfuscate_custom=obfuscate_custom
    )
    return obfuscator.obfuscate(source_code, source_file)
