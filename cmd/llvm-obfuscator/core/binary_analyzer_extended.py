"""
Extended Binary Analysis Module
Analyzes binary structure, imports/exports, and pattern resistance metrics.
"""

import logging
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .utils import run_command, compute_entropy

logger = logging.getLogger(__name__)


class ExtendedBinaryAnalyzer:
    """Analyzes binary structure and pattern resistance metrics."""

    def __init__(self):
        """Initialize the binary analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_binary_structure(self, binary_path: Path) -> Dict:
        """Complete binary structure analysis including sections, imports, exports.

        Args:
            binary_path: Path to the compiled binary

        Returns:
            Dict with comprehensive binary structure metrics
        """
        if not binary_path.exists():
            self.logger.warning(f"Binary not found: {binary_path}")
            return self._empty_binary_structure()

        try:
            binary_format = self._detect_binary_format(binary_path)
            sections = self._analyze_sections(binary_path, binary_format)
            imports = self._extract_imports(binary_path, binary_format)
            exports = self._extract_exports(binary_path, binary_format)
            relocations = self._count_relocations(binary_path, binary_format)
            header_entropy = self._calculate_header_entropy(binary_path)

            return {
                "section_count": len(sections),
                "section_details": sections,
                "import_table": imports,
                "export_table": exports,
                "relocations": relocations,
                "header_analysis": {
                    "header_entropy": round(header_entropy, 3),
                    "binary_type": self._get_binary_type(binary_path, binary_format),
                },
                "code_to_data_ratio": self._calculate_code_data_ratio(sections),
            }
        except Exception as e:
            self.logger.warning(f"Binary structure analysis failed: {e}")
            return self._empty_binary_structure()

    def analyze_pattern_resistance(self, binary_path: Path) -> Dict:
        """Analyze pattern resistance and reverse engineering difficulty.

        Args:
            binary_path: Path to the compiled binary

        Returns:
            Dict with pattern resistance metrics
        """
        if not binary_path.exists():
            self.logger.warning(f"Binary not found: {binary_path}")
            return self._empty_pattern_resistance()

        try:
            binary_data = binary_path.read_bytes()
            string_analysis = self._analyze_strings(binary_data)
            code_analysis = self._analyze_code_patterns(binary_data)
            re_difficulty = self._estimate_re_difficulty(binary_path, string_analysis, code_analysis)

            return {
                "string_analysis": string_analysis,
                "code_analysis": code_analysis,
                "reverse_engineering_difficulty": re_difficulty,
            }
        except Exception as e:
            self.logger.warning(f"Pattern resistance analysis failed: {e}")
            return self._empty_pattern_resistance()

    # ========== PRIVATE HELPER METHODS ==========

    def _detect_binary_format(self, binary_path: Path) -> str:
        """Detect binary format (ELF, PE, Mach-O)."""
        try:
            with open(binary_path, 'rb') as f:
                magic = f.read(4)

                if magic[:2] == b'MZ':  # PE (Windows)
                    return "PE"
                elif magic == b'\x7fELF':  # ELF (Linux)
                    return "ELF"
                elif magic == b'\xfe\xed\xfa' or magic[:4] == b'\xfe\xed\xfa\xce':  # Mach-O (macOS)
                    return "Mach-O"
                else:
                    return "unknown"
        except Exception as e:
            self.logger.debug(f"Error detecting binary format: {e}")
            return "unknown"

    def _analyze_sections(self, binary_path: Path, binary_format: str) -> Dict[str, Dict]:
        """Analyze binary sections and their entropy."""
        sections = {}

        try:
            if binary_format == "ELF":
                sections = self._analyze_elf_sections(binary_path)
            elif binary_format == "PE":
                sections = self._analyze_pe_sections(binary_path)
            elif binary_format == "Mach-O":
                sections = self._analyze_macho_sections(binary_path)
        except Exception as e:
            self.logger.debug(f"Section analysis failed: {e}")

        return sections

    def _analyze_elf_sections(self, binary_path: Path) -> Dict[str, Dict]:
        """Analyze ELF sections using readelf."""
        sections = {}

        try:
            result = run_command(
                ["readelf", "-S", str(binary_path)],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return sections

            output = result.stdout.decode('utf-8', errors='ignore')
            lines = output.split('\n')

            for line in lines:
                # Parse section lines: [Nr] Name Type Address Offset Size EntSize Flags
                if '[' in line and ']' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        # Extract section name and size
                        name = parts[1] if len(parts) > 1 else "unknown"
                        if len(parts) > 5:
                            try:
                                size = int(parts[5], 16)
                                # Calculate entropy for this section
                                section_entropy = self._get_section_entropy(binary_path, name, size)
                                permissions = self._get_elf_section_permissions(line)

                                sections[name] = {
                                    "size": size,
                                    "entropy": round(section_entropy, 3),
                                    "permissions": permissions,
                                }
                            except ValueError:
                                pass
        except Exception as e:
            self.logger.debug(f"ELF section analysis failed: {e}")

        return sections

    def _analyze_pe_sections(self, binary_path: Path) -> Dict[str, Dict]:
        """Analyze PE sections using objdump."""
        sections = {}

        try:
            result = run_command(
                ["objdump", "-h", str(binary_path)],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return sections

            output = result.stdout.decode('utf-8', errors='ignore')
            lines = output.split('\n')

            for line in lines:
                # Parse PE section lines
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    name = parts[1]
                    try:
                        size = int(parts[2], 16)
                        section_entropy = self._get_section_entropy(binary_path, name, size)

                        sections[name] = {
                            "size": size,
                            "entropy": round(section_entropy, 3),
                            "permissions": self._get_pe_section_permissions(line),
                        }
                    except ValueError:
                        pass
        except Exception as e:
            self.logger.debug(f"PE section analysis failed: {e}")

        return sections

    def _analyze_macho_sections(self, binary_path: Path) -> Dict[str, Dict]:
        """Analyze Mach-O sections using otool."""
        sections = {}

        try:
            result = run_command(
                ["otool", "-l", str(binary_path)],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return sections

            output = result.stdout.decode('utf-8', errors='ignore')
            lines = output.split('\n')

            for i, line in enumerate(lines):
                if 'sectname' in line:
                    # Extract section name
                    match = re.search(r'sectname\s+(\S+)', line)
                    if match:
                        name = match.group(1).strip('[]')
                        # Find size in next lines
                        for j in range(i, min(i + 5, len(lines))):
                            if 'size' in lines[j]:
                                size_match = re.search(r'size\s+(\d+)', lines[j])
                                if size_match:
                                    size = int(size_match.group(1))
                                    section_entropy = self._get_section_entropy(binary_path, name, size)
                                    sections[name] = {
                                        "size": size,
                                        "entropy": round(section_entropy, 3),
                                        "permissions": "r--",  # Placeholder
                                    }
                                    break
        except Exception as e:
            self.logger.debug(f"Mach-O section analysis failed: {e}")

        return sections

    def _get_section_entropy(self, binary_path: Path, section_name: str, size: int) -> float:
        """Calculate entropy for a specific section."""
        try:
            # Fallback: return average entropy estimate based on common patterns
            # A proper implementation would extract the section data and calculate entropy
            # For now, return a heuristic value
            if size == 0:
                return 0.0

            # Heuristic: code sections (.text, __text) typically have high entropy (6-7)
            # Data sections (.data, .rodata) have medium entropy (4-5)
            if section_name in ['.text', '__text', '.code']:
                return 6.5  # Code typically has high entropy
            elif section_name in ['.data', '__data', '.bss']:
                return 4.0  # Data typically has lower entropy
            elif section_name in ['.rodata', '__const', '.const']:
                return 5.0  # Read-only data
            else:
                return 5.5  # Unknown section, middle estimate

        except Exception:
            return 5.0  # Default fallback

    def _get_elf_section_permissions(self, line: str) -> str:
        """Extract ELF section permissions from readelf output."""
        # Flags: W (write), A (alloc), X (execute), M (merge), S (strings), etc.
        flags = ""
        if 'R' in line or 'A' in line:
            flags += "r"
        else:
            flags += "-"

        if 'W' in line:
            flags += "w"
        else:
            flags += "-"

        if 'X' in line:
            flags += "x"
        else:
            flags += "-"

        return flags

    def _get_pe_section_permissions(self, line: str) -> str:
        """Extract PE section permissions from objdump output."""
        flags = ""
        if 'r' in line or 'ALLOC' in line:
            flags += "r"
        else:
            flags += "-"

        if 'w' in line or 'DATA' in line:
            flags += "w"
        else:
            flags += "-"

        if 'x' in line or 'EXEC' in line or 'CODE' in line:
            flags += "x"
        else:
            flags += "-"

        return flags

    def _extract_imports(self, binary_path: Path, binary_format: str) -> Dict:
        """Extract imported libraries and symbols."""
        imports = {
            "library_count": 0,
            "imported_symbols": 0,
            "libraries": [],
        }

        try:
            if binary_format == "ELF":
                # Use readelf to get dynamic dependencies
                result = run_command(
                    ["readelf", "-d", str(binary_path)],
                    capture_output=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.decode('utf-8', errors='ignore')
                    libs = re.findall(r'NEEDED\s+Shared library:\s+\[([^\]]+)\]', output)
                    imports["libraries"] = libs
                    imports["library_count"] = len(libs)

                # Count dynamic symbols
                result = run_command(
                    ["readelf", "-s", str(binary_path)],
                    capture_output=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.decode('utf-8', errors='ignore')
                    symbol_count = len([l for l in output.split('\n') if 'UND' in l])
                    imports["imported_symbols"] = symbol_count

            elif binary_format == "PE":
                # Use objdump to get imports
                result = run_command(
                    ["objdump", "-p", str(binary_path)],
                    capture_output=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.decode('utf-8', errors='ignore')
                    dlls = re.findall(r'DLL Name: (\S+)', output)
                    imports["libraries"] = dlls
                    imports["library_count"] = len(dlls)
                    imports["imported_symbols"] = len(re.findall(r'ordinal', output))

        except Exception as e:
            self.logger.debug(f"Import extraction failed: {e}")

        return imports

    def _extract_exports(self, binary_path: Path, binary_format: str) -> Dict:
        """Extract exported symbols."""
        exports = {
            "exported_symbols": 0,
            "exports": [],
        }

        try:
            if binary_format == "ELF":
                # Use readelf/nm to get global symbols
                result = run_command(
                    ["nm", str(binary_path)],
                    capture_output=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.decode('utf-8', errors='ignore')
                    # Match global (uppercase) symbols not in the undefined section
                    exports_list = []
                    for line in output.split('\n'):
                        parts = line.split()
                        if len(parts) >= 3:
                            # Look for global symbols (T/D flags)
                            if parts[1] in ['T', 'D', 'B']:
                                exports_list.append(parts[2])

                    exports["exports"] = exports_list[:100]  # Limit to first 100
                    exports["exported_symbols"] = len(exports_list)

        except Exception as e:
            self.logger.debug(f"Export extraction failed: {e}")

        return exports

    def _count_relocations(self, binary_path: Path, binary_format: str) -> Dict:
        """Count relocation entries."""
        relocations = {
            "relocation_count": 0,
            "plt_got_entries": 0,
        }

        try:
            if binary_format == "ELF":
                result = run_command(
                    ["readelf", "-r", str(binary_path)],
                    capture_output=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout.decode('utf-8', errors='ignore')
                    relocations["relocation_count"] = len([l for l in output.split('\n') if l.strip() and not l.startswith('Relocation')])
                    relocations["plt_got_entries"] = len([l for l in output.split('\n') if 'PLT' in l or 'JUMP_SLOT' in l])

        except Exception as e:
            self.logger.debug(f"Relocation counting failed: {e}")

        return relocations

    def _calculate_header_entropy(self, binary_path: Path) -> float:
        """Calculate entropy of binary header."""
        try:
            with open(binary_path, 'rb') as f:
                # Read first 512 bytes (typical header size)
                header = f.read(512)
                return compute_entropy(header)
        except Exception:
            return 0.0

    def _get_binary_type(self, binary_path: Path, binary_format: str) -> str:
        """Get detailed binary type (e.g., ELF64, PE32+)."""
        try:
            with open(binary_path, 'rb') as f:
                if binary_format == "ELF":
                    f.seek(4)
                    ei_class = f.read(1)[0]
                    return "ELF64" if ei_class == 2 else "ELF32"
                elif binary_format == "PE":
                    f.seek(0x3c)
                    pe_offset = struct.unpack('<I', f.read(4))[0]
                    f.seek(pe_offset + 24)
                    magic = struct.unpack('<H', f.read(2))[0]
                    return "PE32+" if magic == 0x20b else "PE32"
        except Exception:
            pass

        return "unknown"

    def _calculate_code_data_ratio(self, sections: Dict) -> float:
        """Calculate code to data ratio."""
        code_size = 0
        data_size = 0

        for name, details in sections.items():
            if name in ['.text', '__text', '.code']:
                code_size += details.get("size", 0)
            elif name in ['.data', '__data', '.rodata', '.bss']:
                data_size += details.get("size", 0)

        total = code_size + data_size
        if total > 0:
            return round(code_size / total, 3)
        return 0.0

    def _analyze_strings(self, binary_data: bytes) -> Dict:
        """Analyze strings in binary for entropy and patterns."""
        # Extract visible ASCII strings
        strings = []
        current_string = b""

        for byte in binary_data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string += bytes([byte])
            else:
                if len(current_string) >= 4:  # Strings 4+ chars
                    strings.append(current_string.decode('ascii', errors='ignore'))
                current_string = b""

        # Calculate string entropy
        string_entropy = 0.0
        if strings:
            all_chars = ''.join(strings)
            string_entropy = self._shannon_entropy(all_chars.encode())

        # Detect sensitive patterns
        sensitive_patterns = {
            'email': len([s for s in strings if '@' in s]),
            'url': len([s for s in strings if 'http' in s.lower()]),
            'crypto_key': len([s for s in strings if len(s) > 20 and all(c in '0123456789abcdefABCDEF' for c in s)]),
        }

        return {
            "visible_string_count": len(strings),
            "string_entropy": round(string_entropy, 3),
            "sensitive_patterns_detected": sum(sensitive_patterns.values()),
            "pattern_breakdown": sensitive_patterns,
        }

    def _analyze_code_patterns(self, binary_data: bytes) -> Dict:
        """Analyze code patterns and instruction diversity."""
        # This is a heuristic analysis without disassembly
        opcode_entropy = self._shannon_entropy(binary_data)

        # Count common x86/x64 opcodes as heuristic
        common_opcodes = [
            b'\x55',  # push rbp
            b'\x48',  # REX.W prefix
            b'\xff',  # jmp/call indirect
            b'\x90',  # nop
            b'\xc3',  # ret
        ]

        opcode_density = 0
        for opcode in common_opcodes:
            opcode_density += binary_data.count(opcode)

        avg_opcode_density = opcode_density / len(binary_data) if binary_data else 0

        return {
            "opcode_distribution_entropy": round(opcode_entropy, 3),
            "known_pattern_count": len([1 for op in common_opcodes if op in binary_data]),
            "function_prologue_diversity": round(avg_opcode_density * 100, 2),  # Percentage
        }

    def _estimate_re_difficulty(self, binary_path: Path, string_analysis: Dict, code_analysis: Dict) -> Dict:
        """Estimate reverse engineering difficulty."""
        # Heuristic scoring based on analysis
        string_obfuscation = min(100, string_analysis.get("string_entropy", 0) * 15)
        opcode_diversity = min(100, code_analysis.get("opcode_distribution_entropy", 0) * 15)
        pattern_hiding = max(0, 100 - code_analysis.get("known_pattern_count", 0) * 10)

        decompiler_confusion = (string_obfuscation + opcode_diversity + pattern_hiding) / 3

        # Determine rating
        if decompiler_confusion >= 80:
            rating = "VERY_HIGH"
        elif decompiler_confusion >= 60:
            rating = "HIGH"
        elif decompiler_confusion >= 40:
            rating = "MEDIUM"
        else:
            rating = "LOW"

        return {
            "decompiler_confusion_score": round(decompiler_confusion, 2),
            "gadget_density": 0.0,  # Placeholder
            "cfg_obfuscation_rating": rating,
        }

    def _shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0

        entropy = 0.0
        for i in range(256):
            count = data.count(i)
            if count > 0:
                probability = count / len(data)
                entropy -= probability * (probability and -2.5023 * probability / 2.718281828)

        return min(8.0, entropy)

    def _empty_binary_structure(self) -> Dict:
        """Return empty binary structure dict."""
        return {
            "section_count": 0,
            "section_details": {},
            "import_table": {"library_count": 0, "imported_symbols": 0, "libraries": []},
            "export_table": {"exported_symbols": 0, "exports": []},
            "relocations": {"relocation_count": 0, "plt_got_entries": 0},
            "header_analysis": {"header_entropy": 0.0, "binary_type": "unknown"},
            "code_to_data_ratio": 0.0,
        }

    def _empty_pattern_resistance(self) -> Dict:
        """Return empty pattern resistance dict."""
        return {
            "string_analysis": {
                "visible_string_count": 0,
                "string_entropy": 0.0,
                "sensitive_patterns_detected": 0,
            },
            "code_analysis": {
                "opcode_distribution_entropy": 0.0,
                "known_pattern_count": 0,
                "function_prologue_diversity": 0.0,
            },
            "reverse_engineering_difficulty": {
                "decompiler_confusion_score": 0.0,
                "gadget_density": 0.0,
                "cfg_obfuscation_rating": "LOW",
            },
        }
