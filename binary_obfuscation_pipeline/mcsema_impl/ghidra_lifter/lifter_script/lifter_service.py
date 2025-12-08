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
GHIDRA_HOME = os.getenv('GHIDRA_HOME', '/opt/ghidra/ghidra_11.1_PUBLIC')
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
            result = subprocess.run(
                [analyze_headless, '-version'],
                capture_output=True,
                text=True,
                timeout=10
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
        with tempfile.TemporaryDirectory(prefix='ghidra_project_') as project_dir:
            project_name = 'binary_analysis'
            analyze_headless = os.path.join(self.ghidra_home, 'support', 'analyzeHeadless')

            # Build analyzeHeadless command
            cmd = [
                analyze_headless,
                project_dir,                    # Project directory
                project_name,                   # Project name
                '-scriptPath', SCRIPT_PATH,     # Path to custom scripts
                '-postScript', 'export_cfg.py', # Script to run after analysis
                output_cfg_path,                # Output file path
                binary_path,                    # Input binary
            ]

            logger.info(f"Lifting binary: {binary_path}")
            logger.debug(f"Command: {' '.join(cmd)}")

            try:
                # Run Ghidra headless analysis
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for analysis
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
    Lift a Windows PE binary to McSema CFG.

    Expects:
    - file: binary file (multipart/form-data)

    Returns:
    - cfg_file: path to generated .cfg
    - stats: function/block/edge counts
    """

    try:
        # Check if binary file provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No filename'}), 400

        # Save binary to temp location
        binary_path = os.path.join(BINARIES_DIR, file.filename)
        file.save(binary_path)

        logger.info(f"Received binary: {file.filename}")

        # Generate output CFG path
        output_cfg = os.path.join(REPORTS_DIR, file.filename.replace('.exe', '.cfg'))

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


@app.route('/lift/file', methods=['POST'])
def lift_file():
    """
    Alternative endpoint: accept binary file path (for volume-mounted files).

    Expects:
    - binary_path: path to binary on shared volume
    - output_dir: output directory for .cfg

    Returns:
    - cfg_file: path to generated .cfg
    - stats: function/block/edge counts
    """

    try:
        data = request.get_json()

        if not data or 'binary_path' not in data:
            return jsonify({'error': 'Missing binary_path in request'}), 400

        binary_path = data['binary_path']
        output_dir = data.get('output_dir', REPORTS_DIR)

        # Validate binary exists
        if not os.path.exists(binary_path):
            return jsonify({'error': f'Binary not found: {binary_path}'}), 404

        # Generate output CFG path
        binary_name = os.path.basename(binary_path)
        output_cfg = os.path.join(output_dir, binary_name.replace('.exe', '.cfg'))

        logger.info(f"Lifting from path: {binary_path}")

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
