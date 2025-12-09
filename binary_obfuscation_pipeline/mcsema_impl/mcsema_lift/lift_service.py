#!/usr/bin/env python3
"""
McSema Lift Service

Wraps mcsema-lift binary and exposes it as a REST API.

This service:
1. Accepts McSema .cfg protobuf files (from json_to_mcsema.py converter)
2. Runs mcsema-lift to convert CFG to LLVM 9 bitcode
3. Returns the .bc file path or downloads

IMPORTANT: Using LLVM 9 version - LLVM 11 has CallSite bug that causes segfaults.

Endpoints:
- POST /lift - Lift CFG to bitcode
- GET /health - Health check
"""

import os
import sys
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - LLVM 9 paths (LLVM 11 has CallSite bug that causes segfaults)
# Paths for official trailofbits/mcsema:llvm9-ubuntu20.04-amd64 image
MCSEMA_LIFT_PATH = '/opt/trailofbits/bin/mcsema-lift-9.0'
OUTPUT_DIR = '/app/output'
CFG_DIR = '/app/cfg'

# Semantics paths for Remill (required by mcsema-lift)
SEMANTICS_PATHS = [
    '/opt/trailofbits/share/remill/9/semantics',
]
SEMANTICS_SEARCH_PATH = ':'.join(SEMANTICS_PATHS)

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)


class McSemaLifter:
    """
    Wrapper for mcsema-lift-9.0 binary.

    mcsema-lift converts a McSema CFG protobuf file (produced by IDA/Ghidra)
    into LLVM 9 bitcode (.bc).

    IMPORTANT: Using LLVM 9 - LLVM 11 has CallSite API bug that causes segfaults.
    This binary is x86_64 only - must run on amd64 platform.
    """

    def __init__(self):
        self.mcsema_lift = MCSEMA_LIFT_PATH
        self._verify_binary()

    def _verify_binary(self):
        """Verify mcsema-lift binary exists and is executable."""
        if not os.path.exists(self.mcsema_lift):
            raise RuntimeError(f"mcsema-lift not found at {self.mcsema_lift}")

        # Try to get version (this also tests if the binary works)
        try:
            result = subprocess.run(
                [self.mcsema_lift, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            logger.info(f"McSema lift verified: {result.stdout.strip()}")
        except Exception as e:
            # Version flag might not work, but binary exists
            logger.warning(f"Could not verify mcsema-lift version: {e}")

    def lift(self, cfg_path: str, output_bc_path: str, arch: str = 'amd64',
             os_type: str = 'windows') -> dict:
        """
        Lift a McSema CFG protobuf to LLVM bitcode.

        Args:
            cfg_path: Path to .cfg protobuf file
            output_bc_path: Path to write .bc output
            arch: Architecture (amd64, x86, aarch64)
            os_type: Operating system (windows, linux, macos)

        Returns:
            Dictionary with status and results
        """
        result = {
            'success': False,
            'bc_file': None,
            'error': None,
            'stdout': '',
            'stderr': '',
        }

        # Validate inputs
        if not os.path.exists(cfg_path):
            result['error'] = f"CFG file not found: {cfg_path}"
            logger.error(result['error'])
            return result

        # Map architecture names
        arch_map = {
            'amd64': 'amd64',
            'x86_64': 'amd64',
            'x86': 'x86',
            'i386': 'x86',
            'aarch64': 'aarch64',
            'arm64': 'aarch64',
        }
        mcsema_arch = arch_map.get(arch.lower(), 'amd64')

        # Map OS names
        os_map = {
            'windows': 'windows',
            'win': 'windows',
            'linux': 'linux',
            'macos': 'macos',
            'darwin': 'macos',
        }
        mcsema_os = os_map.get(os_type.lower(), 'windows')

        # Build mcsema-lift command
        cmd = [
            self.mcsema_lift,
            '--cfg', cfg_path,
            '--output', output_bc_path,
            '--arch', mcsema_arch,
            '--os', mcsema_os,
            '--semantics_search_paths', SEMANTICS_SEARCH_PATH,
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            # Run mcsema-lift with timeout
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            result['stdout'] = process.stdout
            result['stderr'] = process.stderr

            if process.returncode != 0:
                result['error'] = (
                    f"mcsema-lift failed (exit code {process.returncode})\n"
                    f"stderr: {process.stderr[:1000]}"
                )
                logger.error(result['error'])
                return result

            # Verify output was created
            if not os.path.exists(output_bc_path):
                result['error'] = f"Output .bc file not created: {output_bc_path}"
                logger.error(result['error'])
                return result

            result['success'] = True
            result['bc_file'] = output_bc_path

            bc_size = os.path.getsize(output_bc_path)
            logger.info(f"✓ Lift successful: {output_bc_path} ({bc_size} bytes)")

            return result

        except subprocess.TimeoutExpired:
            result['error'] = "mcsema-lift timeout (10 minutes exceeded)"
            logger.error(result['error'])
            return result
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            logger.error(result['error'], exc_info=True)
            return result


# Initialize lifter
try:
    lifter = McSemaLifter()
    logger.info("McSema lifter initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize McSema lifter: {e}")
    lifter = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if lifter is None:
        return jsonify({
            'status': 'unhealthy',
            'service': 'mcsema-lift',
            'error': 'lifter not initialized'
        }), 503

    return jsonify({
        'status': 'healthy',
        'service': 'mcsema-lift',
        'mcsema_path': MCSEMA_LIFT_PATH
    }), 200


@app.route('/lift', methods=['POST'])
def lift_cfg():
    """
    Lift a McSema CFG to LLVM bitcode.

    Accepts:
    - file: CFG file upload
    OR
    - cfg_path: Path to existing CFG file on shared volume

    Optional parameters:
    - output_path: Output .bc file path
    - arch: Architecture (amd64, x86)
    - os: Operating system (windows, linux)

    Returns:
    - bc_file: Path to generated .bc file
    """
    if lifter is None:
        return jsonify({'error': 'McSema lifter not initialized'}), 503

    try:
        # Get CFG file (either uploaded or path)
        cfg_path = None

        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                cfg_path = os.path.join(CFG_DIR, file.filename)
                file.save(cfg_path)
                logger.info(f"Received CFG file: {file.filename}")
        elif request.is_json:
            data = request.get_json()
            cfg_path = data.get('cfg_path')
        else:
            cfg_path = request.form.get('cfg_path')

        if not cfg_path:
            return jsonify({'error': 'No CFG file provided. Use "file" upload or "cfg_path" parameter.'}), 400

        if not os.path.exists(cfg_path):
            return jsonify({'error': f'CFG file not found: {cfg_path}'}), 404

        # Get optional parameters
        if request.is_json:
            data = request.get_json()
            output_path = data.get('output_path')
            arch = data.get('arch', 'amd64')
            os_type = data.get('os', 'windows')
        else:
            output_path = request.form.get('output_path')
            arch = request.form.get('arch', 'amd64')
            os_type = request.form.get('os', 'windows')

        # Generate output path if not provided
        if not output_path:
            cfg_basename = os.path.basename(cfg_path)
            bc_name = cfg_basename.replace('.cfg', '.bc').replace('.json', '.bc')
            output_path = os.path.join(OUTPUT_DIR, bc_name)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run lift
        result = lifter.lift(cfg_path, output_path, arch, os_type)

        if result['success']:
            return jsonify({
                'success': True,
                'bc_file': result['bc_file'],
                'size': os.path.getsize(result['bc_file']),
                'next_stage': 'READY_FOR_OLLVM',
                'next_action': f'opt -load LLVMObfuscation.so -bcf -fla -sub {result["bc_file"]} -o obfuscated.bc'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

    except Exception as e:
        logger.error(f"Lift endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/lift/file', methods=['POST'])
def lift_file():
    """
    Alternative endpoint: lift a CFG file by path (for volume-mounted files).

    Expects JSON:
    {
        "cfg_path": "/app/cfg/program.cfg",
        "output_path": "/app/output/program.bc",  // optional
        "arch": "amd64",                          // optional, default: amd64
        "os": "windows"                           // optional, default: windows
    }
    """
    if lifter is None:
        return jsonify({'error': 'McSema lifter not initialized'}), 503

    try:
        data = request.get_json()
        if not data or 'cfg_path' not in data:
            return jsonify({'error': 'Missing cfg_path in request'}), 400

        cfg_path = data['cfg_path']
        if not os.path.exists(cfg_path):
            return jsonify({'error': f'CFG not found: {cfg_path}'}), 404

        # Get parameters
        output_path = data.get('output_path')
        if not output_path:
            cfg_basename = os.path.basename(cfg_path)
            bc_name = cfg_basename.replace('.cfg', '.bc').replace('.json', '.bc')
            output_path = os.path.join(OUTPUT_DIR, bc_name)

        arch = data.get('arch', 'amd64')
        os_type = data.get('os', 'windows')

        logger.info(f"Lifting from path: {cfg_path} -> {output_path}")

        # Run lift
        result = lifter.lift(cfg_path, output_path, arch, os_type)

        if result['success']:
            return jsonify({
                'success': True,
                'bc_file': result['bc_file'],
                'size': os.path.getsize(result['bc_file']),
                'next_stage': 'READY_FOR_OLLVM',
                'next_action': f'opt -load LLVMObfuscation.so -bcf -fla -sub {result["bc_file"]} -o obfuscated.bc'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

    except Exception as e:
        logger.error(f"Lift file endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_bc(filename):
    """Download a generated .bc file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {filename}'}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    logger.info("Starting McSema Lift Service on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False)
