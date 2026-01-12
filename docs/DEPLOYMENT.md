# Deployment & Operations

This replaces `DEPLOYMENT_VERIFICATION.md`, `WINDOWS_*` notes, and the GCP helper docs.  It summarizes how to package, validate, and ship the obfuscator.

## 1. Packaging Options

| Option | When to Use | Notes |
|--------|-------------|-------|
| **CLI Distribution** | Local builds, CI jobs | Ensure `pip install -r requirements.txt` and `mlir-obs/build.sh` have been run.  Publish to PyPI once tests pass. |
| **Docker Image** | Reproducible environments, API deployment | Use `docker build -t llvm-obfuscator:latest -f cmd/llvm-obfuscator/Dockerfile.backend .`.  Image bundles LLVM plugins, MLIR library, and the FastAPI server. |
| **Server Mode** | Hosted SaaS style deployment | `uvicorn api.server:app --host 0.0.0.0 --port 4666` launches the backend with WebSocket progress streaming. |
| **Windows Pipeline** | PE-only customers | Run Docker Compose in `binary_obfuscation_pipeline/mcsema_impl` to start the Ghidra lifter and McSema services. |

## 2. Artifact Storage (GCP Example)

Use `gsutil` directly when mirroring SDKs, LLVM plugins, or verification binaries.

```bash
# Upload a new macOS SDK bundle
gsutil cp MacOSX15.4.sdk.tar.gz gs://llvmbins/macos-sdk/MacOSX15.4.sdk.tar.gz

# Download previously uploaded LLVM plugins
gsutil cp gs://llvmbins/plugins/latest.tar.gz ./cmd/llvm-obfuscator/plugins
```

Always version files in the bucket, document SHA256 hashes in release notes, and keep bucket ACLs locked down to maintainers.

## 3. Deployment Verification

- **Linux/macOS** – Run the CLI smoke test and ensure `mlir-obs/test.sh` completes successfully.
- **Windows** – Cross-compile via MinGW (see Stage 1 of the lifting pipeline) or target Windows using `--target x86_64-w64-windows-gnu` from a Linux host.  Verify that UPX packing still produces runnable binaries.
- **API** – Hit `/api/health` and `/api/jobs` endpoints after deployment to confirm the FastAPI instance can spawn jobs and stream progress events.
- **Security Analysis** – Include at least one `phoronix/scripts/run_obfuscation_test_suite.sh` run in the release QA so the protection index reflects the current build.

## 4. Windows Specific Fixes & Tips

- Compile with `-fno-asynchronous-unwind-tables`, `-fno-exceptions`, and `-fno-inline` when preparing binaries for CFG lifting.
- Disable SEH/stack cookies to keep McSema happy.
- Avoid recursion and switch statements; use explicit `if/else` chains for now.
- UPX can produce “bad loader” errors on certain Windows binaries.  If that occurs, re-run without UPX or rebuild with `/MT` disabled.

## 5. Release Template

1. Tag the commit (`git tag vX.Y.Z && git push --tags`).
2. Publish CLI package (`python -m build && twine upload dist/*`).
3. Publish Docker image (`docker push your-registry/llvm-obfuscator:vX.Y.Z`).
4. Upload refreshed plugin archives/SDK bundles to GCP (if applicable).
5. Attach the latest report (JSON/MD) to the GitHub release so users can inspect the metrics.

Keeping these steps documented ensures new contributors—and future GSoC students—have a predictable process for shipping improvements.
