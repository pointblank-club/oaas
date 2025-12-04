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
    ├── upload-binaries-to-gcp.sh      # Upload binaries to GCP
    └── download-binaries-from-gcp.sh  # Download binaries from GCP
```

## GCP Cloud Storage Bucket

- **Bucket Name:** `llvmbins`
- **Project:** `unified-coyote-478817-r3`
- **Location:** Multi-region (US)

### Binaries Stored in GCP

```
gs://llvmbins/
├── llvm-binaries-linux-x86_64-latest.tar.gz
├── llvm-binaries-linux-x86_64-v1.0.0.tar.gz
├── llvm-binaries-linux-x86_64-v1.0.1.tar.gz
└── ...
```

## Workflow

### 1. Upload Binaries to GCP (Local Development)

When you have new LLVM binaries built locally:

```bash
# Upload binaries to GCP with a specific version
./scripts/upload-binaries-to-gcp.sh v1.0.0

# This will:
# - Read binaries from cmd/llvm-obfuscator/plugins/linux-x86_64/
# - Create a tar.gz archive
# - Upload to gs://llvmbins/llvm-binaries-linux-x86_64-v1.0.0.tar.gz
# - Also upload as "latest" version
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
# Download latest binaries
./scripts/download-binaries-from-gcp.sh

# Download specific version
./scripts/download-binaries-from-gcp.sh v1.0.0

# This will:
# - Download from gs://llvmbins/
# - Extract to cmd/llvm-obfuscator/plugins/linux-x86_64/
# - Verify binaries are valid ELF files
```

### 3. CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline automatically handles binaries:

#### DockerHub Build & Push (`.github/workflows/dockerhub-deploy.yml`)

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
# Downloads from GCP
gsutil cp gs://llvmbins/llvm-binaries-linux-x86_64-latest.tar.gz /tmp/
tar -xzf /tmp/llvm-binaries.tar.gz -C cmd/llvm-obfuscator/plugins/

# Verifies binaries
file cmd/llvm-obfuscator/plugins/linux-x86_64/clang  # Should be ELF
file cmd/llvm-obfuscator/plugins/linux-x86_64/opt    # Should be ELF

# Builds Docker images
docker build -f cmd/llvm-obfuscator/Dockerfile.backend

# Pushes to DockerHub
docker push skysingh04/llvm-obfuscator-backend:latest
```

#### Server Deployment (`.github/workflows/deploy.yml`)

```yaml
# Triggered after DockerHub build completes
steps:
  1. SSH to PB server
  2. Pull latest docker-compose.yml from Git
  3. Pull Docker images from DockerHub
  4. Deploy containers
```

**What happens on server:**
```bash
cd /home/devopswale/oaas/
git pull origin main
docker compose pull       # Pulls from DockerHub
docker compose up -d
```

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
   ./scripts/upload-binaries-to-gcp.sh v1.0.1
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

```bash
# SSH to server
ssh user@pb-server

# Download binaries
cd /home/devopswale/oaas
./scripts/download-binaries-from-gcp.sh latest

# Rebuild and redeploy
docker compose build
docker compose up -d
```

## Troubleshooting

### "Binaries missing in CI"

**Error:** CI fails with "clang binary not found"

**Solution:**
```bash
# Upload binaries to GCP first
./scripts/upload-binaries-to-gcp.sh latest

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
./scripts/download-binaries-from-gcp.sh latest
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

## Binary Archive Format

The archive structure is:

```
llvm-binaries-linux-x86_64-v1.0.0.tar.gz
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

## Scripts Reference

### upload-binaries-to-gcp.sh

**Purpose:** Upload local binaries to GCP Cloud Storage

**Usage:**
```bash
./scripts/upload-binaries-to-gcp.sh [version]
```

**Arguments:**
- `version` - Version tag (e.g., v1.0.0), defaults to v1.0.0

**What it does:**
1. Verifies binaries exist in `cmd/llvm-obfuscator/plugins/linux-x86_64/`
2. Verifies binaries are valid ELF files
3. Creates tar.gz archive
4. Uploads to `gs://llvmbins/llvm-binaries-linux-x86_64-${version}.tar.gz`
5. Also uploads as `latest` version

### download-binaries-from-gcp.sh

**Purpose:** Download binaries from GCP Cloud Storage to local machine

**Usage:**
```bash
./scripts/download-binaries-from-gcp.sh [version]
```

**Arguments:**
- `version` - Version to download (e.g., v1.0.0), defaults to "latest"

**What it does:**
1. Downloads archive from GCP
2. Extracts to `cmd/llvm-obfuscator/plugins/linux-x86_64/`
3. Verifies binaries are valid

## Best Practices

1. **Version your binaries**: Always use version tags when uploading
   ```bash
   ./scripts/upload-binaries-to-gcp.sh v1.0.1
   ```

2. **Test locally first**: Download and test binaries before deploying
   ```bash
   ./scripts/download-binaries-from-gcp.sh v1.0.1
   docker compose build
   docker compose up
   ```

3. **Keep binaries out of Git**: Never commit large binaries to Git
   - They're in `.gitignore`
   - Use GCP Cloud Storage instead

4. **Document changes**: When uploading new binaries, document what changed
   ```bash
   # Add to commit message or PR description
   "Updated LLVM binaries to v1.0.1 - Fixed obfuscation plugin crash"
   ```

5. **Backup important versions**: Keep multiple versions in GCP
   ```bash
   # Don't delete old versions unless necessary
   gsutil ls gs://llvmbins/
   ```

## Security Notes

- Service account keys should NEVER be committed to Git
- Store keys securely (GitHub Secrets, environment variables)
- Use least-privilege permissions (only Cloud Storage access)
- Rotate service account keys periodically

## Summary

**For Local Development:**
```bash
# Upload binaries
./scripts/upload-binaries-to-gcp.sh v1.0.1

# Download binaries
./scripts/download-binaries-from-gcp.sh latest
```

**For CI/CD:**
- Push code → CI downloads from GCP → Builds Docker images → Pushes to DockerHub → Server deploys

**For Server:**
- Just pulls Docker images from DockerHub (binaries already included)
