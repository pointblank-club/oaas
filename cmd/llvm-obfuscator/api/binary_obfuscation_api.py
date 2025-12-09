"""
Binary Obfuscation API Endpoints

Provides REST API for the binary-to-binary obfuscation pipeline.
"""

import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from core.binary_pipeline_worker import BinaryPipelineWorker

logger = logging.getLogger(__name__)

# Job tracking store (in-memory; use Redis/DB in production)
JOBS: Dict[str, Dict] = {}
# Use shared volume path accessible by both backend and ghidra-lifter containers
JOBS_DIR = Path("/app/binary_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def register_binary_obfuscation_routes(app: FastAPI):
    """Register binary obfuscation routes to the FastAPI app."""

    @app.post("/api/binary_obfuscate")
    async def binary_obfuscate(
        passes: str = None,
        file: UploadFile = File(...)
    ):
        """
        POST /api/binary_obfuscate

        Start a binary obfuscation job.

        Parameters:
        - file: Windows PE binary (.exe)
        - passes: JSON string with pass configuration

        Returns:
        {
            "job_id": "...",
            "status": "QUEUED"
        }
        """

        try:
            # Validate file extension
            if not file.filename.lower().endswith(".exe"):
                raise HTTPException(
                    status_code=400,
                    detail="Only .exe files are supported"
                )

            # Parse passes configuration
            try:
                passes_config = json.loads(passes) if passes else {"substitution": True}
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid passes JSON"
                )

            # Only substitution is allowed for now
            if not isinstance(passes_config.get("substitution"), bool):
                passes_config = {"substitution": True}

            # Create job directory
            job_id = str(uuid.uuid4())
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded binary
            input_exe = job_dir / "input.exe"
            content = await file.read()
            input_exe.write_bytes(content)

            # Create metadata
            metadata = {
                "job_id": job_id,
                "input_file": file.filename,
                "input_size": len(content),
                "passes_config": passes_config,
                "status": "QUEUED",
                "stage": "GHIDRA",
                "progress": 0
            }

            metadata_file = job_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2))

            # Store job info
            JOBS[job_id] = {
                "job_dir": str(job_dir),
                "status": "QUEUED",
                "stage": "GHIDRA",
                "progress": 0,
                "worker": None
            }

            # Start pipeline in background thread
            def run_pipeline():
                try:
                    JOBS[job_id]["status"] = "RUNNING"
                    JOBS[job_id]["stage"] = "GHIDRA"
                    JOBS[job_id]["progress"] = 10

                    worker = BinaryPipelineWorker(str(job_dir), passes_config)
                    JOBS[job_id]["worker"] = worker

                    success, message = worker.execute()

                    if success:
                        JOBS[job_id]["status"] = "COMPLETED"
                        JOBS[job_id]["stage"] = "COMPLETED"
                        JOBS[job_id]["progress"] = 100

                        # Update metadata
                        metadata["status"] = "COMPLETED"
                        metadata_file.write_text(json.dumps(metadata, indent=2))
                    else:
                        JOBS[job_id]["status"] = "ERROR"
                        JOBS[job_id]["error"] = message
                        logger.error(f"Job {job_id} failed: {message}")

                except Exception as e:
                    JOBS[job_id]["status"] = "ERROR"
                    JOBS[job_id]["error"] = str(e)
                    logger.error(f"Job {job_id} exception: {e}")

            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()

            return JSONResponse({
                "job_id": job_id,
                "status": "QUEUED",
                "message": "Job queued for processing"
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Binary obfuscation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/binary_obfuscate/status/{job_id}")
    async def get_job_status(job_id: str):
        """
        GET /api/binary_obfuscate/status/{job_id}

        Get job status and progress.

        Returns:
        {
            "job_id": "...",
            "status": "QUEUED|RUNNING|COMPLETED|ERROR",
            "stage": "GHIDRA|LIFTING|IR22|OLLVM|FINALIZING|COMPLETED|ERROR",
            "progress": 0-100,
            "logs": "...",
            "metrics": {...},
            "download_url": "..."
        }
        """

        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")

        job = JOBS[job_id]
        job_dir = Path(job["job_dir"])

        # Map internal stage to frontend stage names
        stage_mapping = {
            "GHIDRA": "CFG",
            "LIFTING": "LIFTING",
            "IR22": "IR22",
            "OLLVM": "OLLVM",
            "FINALIZING": "FINALIZING",
            "COMPLETED": "COMPLETED",
            "ERROR": "ERROR"
        }

        # Get logs
        logs = ""
        if job["worker"]:
            logs = job["worker"].get_logs()

        # Get metrics
        metrics = None
        if job["worker"]:
            metrics = job["worker"].get_metrics()

        # Build available artifacts list
        available_artifacts = []
        artifact_checks = {
            "final.exe": job_dir / "final" / "final.exe",
            "program_obf.bc": job_dir / "obf" / "program_obf.bc",
            "program_llvm22.bc": job_dir / "ir" / "program_llvm22.bc",
            "logs.txt": job_dir / "logs.txt",
            "metrics.json": job_dir / "metrics.json",
            "input.cfg": job_dir / "cfg" / "input.cfg",
        }
        for name, path in artifact_checks.items():
            if path.exists():
                available_artifacts.append(name)

        # Build response
        response = {
            "job_id": job_id,
            "status": job["status"],
            "stage": stage_mapping.get(job.get("stage", "GHIDRA"), "UNKNOWN"),
            "progress": job.get("progress", 0),
            "logs": logs[-2000:] if logs else "",  # Last 2000 chars
            "available_artifacts": available_artifacts,
        }

        if metrics:
            response["metrics"] = metrics

        if job["status"] == "COMPLETED":
            # Add download URLs
            final_exe = job_dir / "final" / "final.exe"
            obf_bc = job_dir / "obf" / "program_obf.bc"

            if final_exe.exists():
                response["download_url"] = f"/api/binary_obfuscate/artifact/{job_id}/final.exe"
            if obf_bc.exists():
                response["ir_download_url"] = f"/api/binary_obfuscate/artifact/{job_id}/program_obf.bc"

        elif job["status"] == "ERROR":
            response["error"] = job.get("error", "Unknown error")

        return JSONResponse(response)

    @app.get("/api/binary_obfuscate/artifact/{job_id}/{artifact_name}")
    async def get_artifact(job_id: str, artifact_name: str):
        """
        GET /api/binary_obfuscate/artifact/{job_id}/{artifact_name}

        Download job artifacts.

        Supported artifacts:
        - final.exe
        - program_obf.bc
        - program_llvm22.bc
        - logs.txt
        - metrics.json
        """

        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")

        job_dir = Path(JOBS[job_id]["job_dir"])

        # Validate artifact name
        allowed_artifacts = {
            "final.exe": job_dir / "final" / "final.exe",
            "program_obf.bc": job_dir / "obf" / "program_obf.bc",
            "program_llvm22.bc": job_dir / "ir" / "program_llvm22.bc",
            "logs.txt": job_dir / "logs.txt",
            "metrics.json": job_dir / "metrics.json",
            "input.cfg": job_dir / "cfg" / "input.cfg",  # Ghidra CFG JSON
        }

        if artifact_name not in allowed_artifacts:
            raise HTTPException(status_code=400, detail="Invalid artifact name")

        artifact_path = allowed_artifacts[artifact_name]

        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_name} not found")

        return FileResponse(
            path=artifact_path,
            filename=artifact_name,
            media_type="application/octet-stream"
        )

    logger.info("Binary obfuscation routes registered")
