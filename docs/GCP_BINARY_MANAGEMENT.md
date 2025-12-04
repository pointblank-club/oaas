# GCP Binary Management for OAAS

This document explains how binaries are managed using Google Cloud Platform (GCP) Cloud Storage for the OAAS project.

## Overview

The OAAS project uses custom-built LLVM binaries (clang, opt, plugins) that are:
1. Built locally or in a build environment
2. Uploaded to GCP Cloud Storage bucket
3. Downloaded during CI/CD to build Docker images
4. Included in Docker images pushed to DockerHub
5. Deployed on the PB server via DockerHub images

**Important:** The `binaries/` folder is NOT tracked by Git (it's in `.gitignore`).

## Directory Structure

```
oaas/
├── binaries/                          # Local binaries (NOT in Git)
│   ├── clang
│   ├── opt
│   ├── mlir-opt
│   ├── LLVMObfuscationPlugin.so
│   └── lib/
├── cmd/llvm-obfuscator/
│   └── plugins/
│       └── linux-x86_64/              # Binaries for Docker builds
│           ├── clang
│           ├── opt
│           └── LLVMObfuscationPlugin.so
└── scripts/
    ├── upload-binaries-to-gcp.sh      # Upload binaries to GCP (individual files)
    ├── download-binaries-from-gcp.sh # Download binaries from GCP (individual files)
    └── download-binaries-server.sh    # Download binaries on server (legacy tar.gz)
```

## GCP Cloud Storage Bucket

- **Bucket Name:** `llvmbins`
- **Project:** `unified-coyote-478817-r3`
- **Location:** Multi-region (US)

### Binaries Stored in GCP

The binaries are stored as **individual files** (not as tar.gz archives) in the bucket:

```
gs://llvmbins/
└── linux-x86_64/
    ├── clang                         # Main compiler
    ├── opt                           # LLVM optimizer
    ├── mlir-opt                      # MLIR optimizer
    ├── mlir-translate                # MLIR translator
    ├── clangir                       # ClangIR tool
    ├── LLVMObfuscationPlugin.so      # Obfuscation plugin
    ├── MLIRObfuscation.so            # MLIR obfuscation plugin
    ├── libLLVM.so.22.0git            # LLVM shared library
    └── lib/
        └── clang/
            └── 22/
                └── include/          # Clang headers
```

**Note:** Some legacy workflows may still reference tar.gz archives, but the primary storage format is individual files.

## Workflow

### 1. Upload Binaries to GCP (Local Development)

When you have new LLVM binaries built locally:

```bash
# Upload all binaries found locally to GCP
./scripts/upload-binaries-to-gcp.sh

# Upload a specific binary only
./scripts/upload-binaries-to-gcp.sh LLVMObfuscationPlugin.so

# This will:
# - Read binaries from cmd/llvm-obfuscator/plugins/linux-x86_64/
# - Verify each binary is a valid ELF file
# - Upload individual files to gs://llvmbins/linux-x86_64/
# - Only uploads binaries that exist locally (doesn't overwrite others)
# - Also uploads clang headers from lib/clang/22/include/ if present
```

**Prerequisites:**
- Binaries must exist in `cmd/llvm-obfuscator/plugins/linux-x86_64/`
- GCP CLI (`gcloud`, `gsutil`) must be installed
- Must be authenticated with GCP:
  ```bash
  # Option 1: Service account
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
  
  # Option 2: User account
  gcloud auth login
  ```

### 2. Download Binaries from GCP (Local Development)

To download binaries from GCP to your local machine:

```bash
# Download all binaries from GCP
./scripts/download-binaries-from-gcp.sh

# This will:
# - Download individual files from gs://llvmbins/linux-x86_64/
# - Save to cmd/llvm-obfuscator/plugins/linux-x86_64/
# - Make binaries executable
# - Verify binaries are valid ELF files
```

**Note:** This script always downloads the latest binaries (no version parameter). The binaries are stored as individual files, not as archives.

### 3. CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline automatically handles binaries. There are multiple workflows that use binaries:

#### DockerHub Build & Push (`.github/workflows/dockerhub-deploy.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Push of tags matching `v*` pattern

**Binary Format:** Individual files (primary format)

```yaml
# Automatically triggered on push to main
steps:
  1. Authenticate with GCP using service account
  2. Download binaries from GCP Cloud Storage
  3. Extract to cmd/llvm-obfuscator/plugins/linux-x86_64/
  4. Verify binaries are valid
  5. Build Docker images with binaries
  6. Push images to DockerHub
```

**What happens:**
```bash
# Downloads individual binaries from GCP
gsutil -m cp -r gs://llvmbins/linux-x86_64/* cmd/llvm-obfuscator/plugins/linux-x86_64/

# Makes binaries executable
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/clang
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/opt
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/mlir-opt
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/mlir-translate
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/clangir

# Verifies binaries
file cmd/llvm-obfuscator/plugins/linux-x86_64/clang  # Should be ELF
file cmd/llvm-obfuscator/plugins/linux-x86_64/opt    # Should be ELF
file cmd/llvm-obfuscator/plugins/linux-x86_64/LLVMObfuscationPlugin.so  # Should be shared object

# Builds Docker images
docker build -f cmd/llvm-obfuscator/Dockerfile.backend

# Pushes to DockerHub
docker push skysingh04/llvm-obfuscator-backend:latest
```

**Required binaries:**
- `clang` - Main compiler
- `opt` - LLVM optimizer
- `LLVMObfuscationPlugin.so` - Obfuscation plugin
- `libLLVM.so.22.0git` - LLVM shared library
- `mlir-opt`, `mlir-translate`, `clangir` - MLIR tools
- `MLIRObfuscation.so` - MLIR obfuscation plugin
- `lib/clang/22/include/` - Clang headers directory

#### Server Deployment (`.github/workflows/deploy.yml`)

```yaml
# Triggered after DockerHub build completes (on success)
steps:
  1. SSH to PB server
  2. Navigate to deployment directory
  3. Pull latest Docker images from DockerHub
  4. Stop existing containers
  5. Start new containers with updated images
  6. Clean up old images
```

**What happens on server:**
```bash
cd /home/devopswale/oaas/
# docker-compose.yml is already in the repository
docker compose pull       # Pulls latest images from DockerHub
docker compose down --remove-orphans
docker compose up -d --remove-orphans
docker image prune -f    # Clean up old images
```

**Triggers:**
- Manual workflow dispatch
- Automatically after successful "Build and Push to Docker Hub" workflow (on `main` branch only)

**Note:** Binaries are already included in the Docker images, so no separate binary download is needed during deployment.

#### Jotai Tests Workflow (`.github/workflows/jotai-tests.yml`)

This workflow uses a **legacy tar.gz archive format** for downloading binaries:

```yaml
# Downloads from tar.gz archive (legacy format)
gsutil cp gs://llvmbins/llvm-binaries-linux-x86_64-latest.tar.gz /tmp/
tar -xzf /tmp/llvm-binaries.tar.gz -C cmd/llvm-obfuscator/plugins/
```

**Triggers:**
- Push to any branch
- Pull requests
- Manual workflow dispatch

**Binary Format:** Tar.gz archive (legacy format)

**Note:** This workflow uses the legacy tar.gz archive format. It may need to be updated to use individual files in the future for consistency.

## Authentication

### Local Development

You need GCP credentials to upload/download binaries:

```bash
# Service account (recommended)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project unified-coyote-478817-r3

# User account
gcloud auth login
gcloud config set project unified-coyote-478817-r3
```

### CI/CD (GitHub Actions)

Uses GitHub Secrets:
- `GCP_SERVICE_ACCOUNT_KEY` - Service account JSON key with Cloud Storage permissions

## Updating Binaries

### Standard Update Flow

1. **Build new binaries locally** (or receive them from build server)
   ```bash
   # Place binaries in cmd/llvm-obfuscator/plugins/linux-x86_64/
   ```

2. **Upload to GCP**
   ```bash
   # Upload all binaries
   ./scripts/upload-binaries-to-gcp.sh
   
   # Or upload a specific binary
   ./scripts/upload-binaries-to-gcp.sh clang
   ```

3. **Trigger CI/CD** (or it will run automatically on next push)
   ```bash
   git push origin main
   # This triggers DockerHub build which downloads from GCP
   ```

4. **Deploy** (automatic after DockerHub build)
   ```
   Server automatically pulls new images from DockerHub
   ```

### Emergency Update (Direct Server Update)

If you need to update binaries on the server without rebuilding Docker images:

**Option 1: Use server download script (legacy tar.gz format)**
```bash
# SSH to server
ssh user@pb-server

# Download binaries using server script (uses tar.gz archive)
cd /home/devopswale/oaas
./scripts/download-binaries-server.sh latest

# Rebuild and redeploy
docker compose build
docker compose up -d
```

**Option 2: Download individual files (recommended)**
```bash
# SSH to server
ssh user@pb-server

# Download individual binaries
cd /home/devopswale/oaas
gsutil -m cp -r gs://llvmbins/linux-x86_64/* cmd/llvm-obfuscator/plugins/linux-x86_64/

# Rebuild and redeploy
docker compose build
docker compose up -d
```

**Note:** The `download-binaries-server.sh` script uses the legacy tar.gz archive format. For consistency, prefer downloading individual files directly.

## Troubleshooting

### "Binaries missing in CI"

**Error:** CI fails with "clang binary not found"

**Solution:**
```bash
# Upload binaries to GCP first
./scripts/upload-binaries-to-gcp.sh

# Verify binaries are in GCP
gsutil ls gs://llvmbins/linux-x86_64/

# Then retry CI
```

### "Authentication failed"

**Error:** "Not authenticated with GCP"

**Solution:**
```bash
# Check authentication
gcloud auth list

# Re-authenticate
gcloud auth login
# or
gcloud auth activate-service-account --key-file=/path/to/key.json
```

### "Binary verification failed"

**Error:** "clang exists but is not a valid ELF binary"

**Cause:** Git LFS pointer file instead of actual binary

**Solution:**
```bash
# Make sure you have the actual binary files
file cmd/llvm-obfuscator/plugins/linux-x86_64/clang
# Should show: "ELF 64-bit LSB executable"
# Not: "ASCII text" (which would be a Git LFS pointer)

# If it's a pointer, download from GCP
./scripts/download-binaries-from-gcp.sh
```

### "Upload failed"

**Error:** "Failed to upload binaries to GCP"

**Solution:**
```bash
# Check bucket permissions
gsutil ls gs://llvmbins/

# Check if bucket exists
gsutil mb gs://llvmbins/  # Create if needed

# Check service account permissions
# Need: Storage Object Admin or Storage Object Creator
```

## Binary Storage Format

The project uses **two storage formats** for backward compatibility:

### Primary Format: Individual Files (Recommended)

**Used by:**
- `upload-binaries-to-gcp.sh` script
- `download-binaries-from-gcp.sh` script
- `dockerhub-deploy.yml` workflow

Binaries are stored as **individual files** in GCP Cloud Storage:

```
gs://llvmbins/linux-x86_64/
├── clang                         # Main compiler
├── opt                           # LLVM optimizer
├── mlir-opt                      # MLIR optimizer
├── mlir-translate                # MLIR translator
├── clangir                       # ClangIR tool
├── LLVMObfuscationPlugin.so      # Obfuscation plugin
├── MLIRObfuscation.so            # MLIR obfuscation plugin
├── libLLVM.so.22.0git            # LLVM shared library
└── lib/
    └── clang/
        └── 22/
            └── include/          # Clang headers
```

### Legacy Format: Tar.gz Archives

**Used by:**
- `jotai-tests.yml` workflow
- `download-binaries-server.sh` script

Some workflows still use tar.gz archives for backward compatibility:

```
llvm-binaries-linux-x86_64-latest.tar.gz
└── linux-x86_64/
    ├── clang
    ├── opt
    └── ... (same structure as above)
```

**Note:** The primary storage format is individual files. Tar.gz archives may exist for backward compatibility but are not the recommended format for new uploads.

### Format Comparison

| Feature | Individual Files | Tar.gz Archives |
|---------|-----------------|-----------------|
| Upload script | `upload-binaries-to-gcp.sh` | N/A (legacy) |
| Download script | `download-binaries-from-gcp.sh` | `download-binaries-server.sh` |
| CI/CD workflow | `dockerhub-deploy.yml` | `jotai-tests.yml` |
| Incremental updates | ✅ Yes (upload specific files) | ❌ No (full archive) |
| Storage efficiency | ✅ Better (only changed files) | ❌ Less efficient |
| Recommended | ✅ **Yes** | ❌ Legacy only |

## Scripts Reference

### upload-binaries-to-gcp.sh

**Purpose:** Upload local binaries to GCP Cloud Storage as individual files

**Usage:**
```bash
# Upload all binaries found locally
./scripts/upload-binaries-to-gcp.sh

# Upload a specific binary only
./scripts/upload-binaries-to-gcp.sh [binary_name]
```

**Arguments:**
- `binary_name` (optional) - Specific binary to upload (e.g., `clang`, `LLVMObfuscationPlugin.so`)
  - If not provided, uploads all binaries found locally

**What it does:**
1. Verifies binaries exist in `cmd/llvm-obfuscator/plugins/linux-x86_64/`
2. Verifies each binary is a valid ELF file or shared object
3. Uploads individual files to `gs://llvmbins/linux-x86_64/`
4. Only uploads binaries that exist locally (doesn't overwrite others)
5. Also uploads clang headers from `lib/clang/22/include/` if present

**Supported binaries:**
- `clang`, `opt`, `mlir-opt`, `mlir-translate`, `clangir`
- `LLVMObfuscationPlugin.so`, `MLIRObfuscation.so`
- `libLLVM.so.22.0git`
- `lib/clang/22/include/` (directory)

### download-binaries-from-gcp.sh

**Purpose:** Download individual binaries from GCP Cloud Storage to local machine

**Usage:**
```bash
./scripts/download-binaries-from-gcp.sh
```

**Arguments:**
- None (always downloads latest individual files)

**What it does:**
1. Downloads individual files from `gs://llvmbins/linux-x86_64/`
2. Saves to `cmd/llvm-obfuscator/plugins/linux-x86_64/`
3. Makes binaries executable (`chmod +x`)
4. Verifies binaries are valid ELF files or shared objects

### download-binaries-server.sh

**Purpose:** Download binaries from GCP Cloud Storage on the server (legacy tar.gz format)

**Usage:**
```bash
./scripts/download-binaries-server.sh [version]
```

**Arguments:**
- `version` - Version to download (e.g., v1.0.0), defaults to "latest"

**What it does:**
1. Downloads tar.gz archive from `gs://llvmbins/llvm-binaries-linux-x86_64-${version}.tar.gz`
2. Extracts to `/app/plugins/linux-x86_64/` (server path)
3. Verifies binaries are valid

**Note:** This script uses the legacy tar.gz archive format. For consistency, prefer using individual file downloads.

## Best Practices

1. **Upload individual binaries**: The primary format is individual files, not tar.gz archives
   ```bash
   # Upload all binaries
   ./scripts/upload-binaries-to-gcp.sh
   
   # Or upload specific binaries as needed
   ./scripts/upload-binaries-to-gcp.sh clang
   ./scripts/upload-binaries-to-gcp.sh LLVMObfuscationPlugin.so
   ```

2. **Test locally first**: Download and test binaries before deploying
   ```bash
   ./scripts/download-binaries-from-gcp.sh
   docker compose build
   docker compose up
   ```

3. **Incremental updates**: You can upload individual binaries without affecting others
   ```bash
   # Only update the plugin, leave other binaries unchanged
   ./scripts/upload-binaries-to-gcp.sh LLVMObfuscationPlugin.so
   ```

3. **Keep binaries out of Git**: Never commit large binaries to Git
   - They're in `.gitignore`
   - Use GCP Cloud Storage instead

4. **Document changes**: When uploading new binaries, document what changed
   ```bash
   # Add to commit message or PR description
   "Updated LLVM binaries to v1.0.1 - Fixed obfuscation plugin crash"
   ```

5. **Check what's in GCP**: List binaries to see what's available
   ```bash
   # List all binaries in GCP
   gsutil ls -lh gs://llvmbins/linux-x86_64/
   
   # Check specific binary
   gsutil ls gs://llvmbins/linux-x86_64/clang
   ```

6. **Verify before uploading**: Make sure binaries are valid before uploading
   ```bash
   # Check binary type
   file cmd/llvm-obfuscator/plugins/linux-x86_64/clang
   # Should show: "ELF 64-bit LSB executable"
   ```

## Security Notes

- Service account keys should NEVER be committed to Git
- Store keys securely (GitHub Secrets, environment variables)
- Use least-privilege permissions (only Cloud Storage access)
- Rotate service account keys periodically

## Summary

**For Local Development:**
```bash
# Upload binaries (all or specific)
./scripts/upload-binaries-to-gcp.sh
./scripts/upload-binaries-to-gcp.sh clang

# Download binaries
./scripts/download-binaries-from-gcp.sh
```

**For CI/CD:**
- Push code → CI downloads from GCP → Builds Docker images → Pushes to DockerHub → Server deploys

**For Server:**
- Just pulls Docker images from DockerHub (binaries already included)
