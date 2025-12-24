#!/usr/bin/env python3
"""
Ghidra Lifter Service
Wraps Ghidra headless execution and exposes it as a service.

This service:
1. Listens on port 5000 for lift requests
2. Accepts Windows PE binaries via HTTP
3. Calls Ghidra analyzeHeadless with export_cfg.py script
4. Returns the McSema .cfg file

Communication model:
- Backend service (llvm-obfuscator-backend) calls this service
- Files are exchanged via shared volumes or HTTP
"""

import os
import sys
import json
import subprocess
import logging
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
GHIDRA_HOME = os.getenv('GHIDRA_HOME', '/opt/ghidra/ghidra_11.2.1_PUBLIC')
# IMPORTANT: Ghidra 11.2.1 requires Java 21. Always use Java 21 path, regardless
# of container's JAVA_HOME env var (which may be incorrectly set to Java 17)
JAVA_HOME = '/usr/lib/jvm/java-21-openjdk'
SCRIPT_PATH = '/app/lifter_script'
REPORTS_DIR = '/app/reports'
BINARIES_DIR = '/app/binaries'

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(BINARIES_DIR, exist_ok=True)


class GhidraLifter:
    """
    Wrapper for Ghidra headless CFG export.

    WHY HEADLESS MODE?
    ==================
    - Backend container has no GUI (headless server)
    - analyzeHeadless allows fully automated analysis
    - No user interaction needed
    - Can be called from Python subprocess
    - Suitable for batch processing

    WHY GHIDRA (not IDA)?
    ====================
    - IDA Pro is proprietary and expensive
    - Ghidra is open-source and maintained by NSA
    - Works well for simple -O0 -g binaries (Feature #1 output)
    - Sufficient for proof-of-concept pipeline
    - Can be extended with custom scripts

    LIMITATIONS:
    ============
    - Function detection less accurate than IDA
    - Switch table recovery unreliable
    - No exception handling support
    - Noisy CFG for complex binaries
    """

    def __init__(self):
        self.ghidra_home = GHIDRA_HOME
        self._verify_ghidra()

    def _verify_ghidra(self):
        """Verify Ghidra is properly installed."""
        analyze_headless = os.path.join(self.ghidra_home, 'support', 'analyzeHeadless')

        if not os.path.exists(analyze_headless):
            raise RuntimeError(
                f"Ghidra analyzeHeadless not found at {analyze_headless}. "
                f"Set GHIDRA_HOME environment variable."
            )

        try:
            # Set JAVA_HOME for Ghidra verification
            env = os.environ.copy()
            env['JAVA_HOME'] = JAVA_HOME

            result = subprocess.run(
                [analyze_headless, '-version'],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            logger.info(f"Ghidra verified: {result.stdout.strip()}")
        except Exception as e:
            raise RuntimeError(f"Failed to verify Ghidra: {e}")

    def lift(self, binary_path: str, output_cfg_path: str) -> dict:
        """
        Lift a Windows PE binary to McSema CFG.

        Args:
            binary_path: Path to .exe file
            output_cfg_path: Path to write .cfg output

        Returns:
            Dictionary with status and results
        """

        result = {
            'success': False,
            'cfg_file': None,
            'error': None,
            'stats': {}
        }

        # Validate inputs
        if not os.path.exists(binary_path):
            result['error'] = f"Binary not found: {binary_path}"
            logger.error(result['error'])
            return result

        # Create Ghidra project directory (temporary)
        project_root = "/app/ghidra_projects"
        os.makedirs(project_root, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix='ghidra_project_', dir=project_root) as project_dir:
            project_name = 'binary_analysis'
            analyze_headless = os.path.join(self.ghidra_home, 'support', 'analyzeHeadless')

            # Build analyzeHeadless command
            cmd = [
                analyze_headless,
                project_dir,                    # Project directory
                project_name,                   # Project name
                '-import', binary_path,         # Input binary
                '-scriptPath', SCRIPT_PATH,     # Path to custom scripts
                '-postScript', 'export_cfg.py', # Script to run after analysis
                output_cfg_path,                # Output file path for script
            ]

            logger.info(f"Lifting binary: {binary_path}")
            logger.debug(f"Command: {' '.join(cmd)}")

            try:
                # Run Ghidra headless analysis with JAVA_HOME set
                env = os.environ.copy()
                env['JAVA_HOME'] = JAVA_HOME

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for analysis
                    env=env
                )

                # Check result
                if process.returncode != 0:
                    result['error'] = (
                        f"Ghidra analysis failed (exit code {process.returncode})\n"
                        f"stderr: {process.stderr[:500]}"
                    )
                    logger.error(result['error'])
                    return result

                # Verify output CFG was created
                if not os.path.exists(output_cfg_path):
                    result['error'] = f"CFG file not created: {output_cfg_path}"
                    logger.error(result['error'])
                    return result

                # Parse CFG to get stats
                try:
                    with open(output_cfg_path, 'r') as f:
                        cfg = json.load(f)
                    result['stats'] = {
                        'functions': len(cfg.get('functions', [])),
                        'total_blocks': sum(
                            len(f.get('basic_blocks', []))
                            for f in cfg.get('functions', [])
                        ),
                        'total_edges': sum(
                            len(f.get('edges', []))
                            for f in cfg.get('functions', [])
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Could not parse CFG stats: {e}")

                result['success'] = True
                result['cfg_file'] = output_cfg_path

                logger.info(
                    f"✓ CFG lifted successfully: "
                    f"{result['stats'].get('functions', 0)} functions, "
                    f"{result['stats'].get('total_blocks', 0)} blocks"
                )

                return result

            except subprocess.TimeoutExpired:
                result['error'] = "Ghidra analysis timeout (5 minutes exceeded)"
                logger.error(result['error'])
                return result
            except Exception as e:
                result['error'] = f"Unexpected error: {str(e)}"
                logger.error(result['error'], exc_info=True)
                return result


# Initialize lifter
try:
    lifter = GhidraLifter()
    logger.info("Ghidra lifter initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Ghidra lifter: {e}")
    sys.exit(1)


# Flask routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ghidra-lifter'}), 200


@app.route('/lift', methods=['POST'])
def lift_binary():
    """
    Lift a Windows PE binary to McSema CFG from a file upload.
    Accepts 'output_cfg_path' in the form data to specify output location.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No filename'}), 400

        # Save binary to a shared location inside the container
        binary_path = os.path.join(BINARIES_DIR, file.filename)
        file.save(binary_path)
        logger.info(f"Received binary: {file.filename}")

        # Prioritize output path from form data; fall back to default if not provided
        output_cfg = request.form.get('output_cfg_path')
        if not output_cfg:
            output_cfg = os.path.join(REPORTS_DIR, file.filename.replace('.exe', '.cfg'))
        
        logger.info(f"Output path set to: {output_cfg}")

        # Perform lifting
        result = lifter.lift(binary_path, output_cfg)

        if result['success']:
            return jsonify({
                'success': True,
                'cfg_file': output_cfg,
                'stats': result['stats'],
                'next_stage': 'READY_FOR_MCSEMA_LIFT',
                'next_action': f'mcsema-lift --cfg {output_cfg} --output program.bc'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

    except Exception as e:
        logger.error(f"Lift endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/convert/mcsema', methods=['POST'])
def convert_to_mcsema():
    """
    Convert Ghidra JSON CFG to McSema protobuf format.

    Expects JSON:
    {
        "json_cfg_path": "/app/reports/program.json",
        "output_cfg_path": "/app/reports/program.cfg",  // optional
        "arch": "amd64"                                 // optional, default: amd64
    }

    Returns:
    - cfg_file: path to generated McSema .cfg protobuf
    - stats: conversion statistics
    """
    try:
        data = request.get_json()
        if not data or 'json_cfg_path' not in data:
            return jsonify({'error': 'Missing json_cfg_path in request'}), 400

        json_cfg_path = data['json_cfg_path']
        if not os.path.exists(json_cfg_path):
            return jsonify({'error': f'JSON CFG not found: {json_cfg_path}'}), 404

        # Output path (default: replace .json with .cfg)
        output_cfg = data.get('output_cfg_path')
        if not output_cfg:
            output_cfg = json_cfg_path.replace('.json', '.cfg')
            if output_cfg == json_cfg_path:
                output_cfg = json_cfg_path + '.cfg'

        arch = data.get('arch', 'amd64')
        os_type = data.get('os', 'windows')

        logger.info(f"Converting JSON CFG to McSema protobuf: {json_cfg_path} -> {output_cfg}")

        # Import and run the converter
        try:
            from json_to_mcsema import convert_json_to_mcsema
            stats = convert_json_to_mcsema(json_cfg_path, output_cfg, arch, os_type)

            return jsonify({
                'success': True,
                'cfg_file': output_cfg,
                'stats': stats,
                'next_stage': 'READY_FOR_MCSEMA_LIFT',
                'next_action': f'mcsema-lift-11.0 --cfg {output_cfg} --output program.bc --arch {arch} --os {os_type}'
            }), 200

        except ImportError as e:
            logger.error(f"Could not import json_to_mcsema: {e}")
            return jsonify({
                'success': False,
                'error': f'Converter module not available: {e}'
            }), 500

    except Exception as e:
        logger.error(f"Convert endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/lift/full', methods=['POST'])
def lift_full_pipeline():
    """
    Full pipeline: Binary -> Ghidra JSON CFG -> McSema protobuf CFG.

    Combines /lift/file and /convert/mcsema into a single call.

    Expects JSON:
    {
        "binary_path": "/app/binaries/program.exe",
        "output_dir": "/app/reports",               // optional
        "arch": "amd64"                             // optional
    }

    Returns:
    - json_cfg_file: path to intermediate JSON CFG
    - mcsema_cfg_file: path to final McSema .cfg protobuf
    - stats: combined statistics
    """
    try:
        data = request.get_json()
        if not data or 'binary_path' not in data:
            return jsonify({'error': 'Missing binary_path in request'}), 400

        binary_path = data['binary_path']
        if not os.path.exists(binary_path):
            return jsonify({'error': f'Binary not found: {binary_path}'}), 404

        output_dir = data.get('output_dir', REPORTS_DIR)
        arch = data.get('arch', 'amd64')
        os_type = data.get('os', 'windows')
        os.makedirs(output_dir, exist_ok=True)

        binary_name = os.path.basename(binary_path)
        base_name = binary_name.replace('.exe', '').replace('.dll', '')

        # Stage 1: Ghidra CFG Export (JSON)
        json_cfg_path = os.path.join(output_dir, f"{base_name}_ghidra.json")
        logger.info(f"Stage 1: Ghidra CFG Export -> {json_cfg_path}")

        lift_result = lifter.lift(binary_path, json_cfg_path)
        if not lift_result['success']:
            return jsonify({
                'success': False,
                'error': f"Stage 1 (Ghidra CFG) failed: {lift_result['error']}"
            }), 500

        # Stage 2: Convert to McSema protobuf
        mcsema_cfg_path = os.path.join(output_dir, f"{base_name}_mcsema.cfg")
        logger.info(f"Stage 2: McSema Protobuf Convert -> {mcsema_cfg_path}")

        try:
            from json_to_mcsema import convert_json_to_mcsema
            convert_stats = convert_json_to_mcsema(json_cfg_path, mcsema_cfg_path, arch, os_type)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f"Stage 2 (McSema Convert) failed: {e}"
            }), 500

        return jsonify({
            'success': True,
            'json_cfg_file': json_cfg_path,
            'mcsema_cfg_file': mcsema_cfg_path,
            'ghidra_stats': lift_result['stats'],
            'mcsema_stats': convert_stats,
            'next_stage': 'READY_FOR_MCSEMA_LIFT',
            'next_action': f'mcsema-lift-11.0 --cfg {mcsema_cfg_path} --output {base_name}.bc --arch {arch} --os {os_type}'
        }), 200

    except Exception as e:
        logger.error(f"Full pipeline endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/lift/file', methods=['POST'])
def lift_file():
    """
    Alternative endpoint: accept binary file path (for volume-mounted files).

    Expects:
    - binary_path: path to binary on shared volume
    - output_cfg_path: optional full path for the output cfg file.

    Returns:
    - cfg_file: path to generated .cfg
    - stats: function/block/edge counts
    """
    try:
        data = request.get_json()
        if not data or 'binary_path' not in data:
            return jsonify({'error': 'Missing binary_path in request'}), 400

        binary_path = data['binary_path']
        if not os.path.exists(binary_path):
            return jsonify({'error': f'Binary not found: {binary_path}'}), 404

        # Prioritize full output path from request; fall back to old logic if not provided
        output_cfg = data.get('output_cfg_path')
        if not output_cfg:
            output_dir = data.get('output_dir', REPORTS_DIR)
            binary_name = os.path.basename(binary_path)
            output_cfg = os.path.join(output_dir, binary_name.replace('.exe', '.cfg'))

        logger.info(f"Lifting from path: {binary_path} -> {output_cfg}")

        # Perform lifting
        result = lifter.lift(binary_path, output_cfg)

        if result['success']:
            return jsonify({
                'success': True,
                'cfg_file': output_cfg,
                'stats': result['stats'],
                'next_stage': 'READY_FOR_MCSEMA_LIFT',
                'next_action': f'mcsema-lift --cfg {output_cfg} --output program.bc'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

    except Exception as e:
        logger.error(f"Lift file endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Ghidra Lifter Service on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
