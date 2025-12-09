# Custom UPX Binary - Docker Integration Guide

## Overview

This guide explains how to integrate your custom UPX binary into the Docker build and CI/CD pipeline.

## Problem

The custom UPX binary (`/home/dhruv/Documents/upx/build/upx`) exists only on your local machine. When Docker Hub builds the image, it needs access to this binary. We solve this by:

1. **Uploading the UPX binary to GCP** (same storage as other LLVM binaries)
2. **Downloading it during CI/CD** (along with other binaries)
3. **Including it in the Docker image** (at `/usr/local/llvm-obfuscator/bin/upx`)

## Step 1: Upload UPX Binary to GCP

First, copy your UPX binary to the plugins directory:

```bash
cd /home/dhruv/Documents/Code/oaas
mkdir -p cmd/llvm-obfuscator/plugins/linux-x86_64
cp /home/dhruv/Documents/upx/build/upx cmd/llvm-obfuscator/plugins/linux-x86_64/upx
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/upx
```

Then upload it to GCP using the binary manager:

```bash
./scripts/gcp-binary-manager.sh add \
    cmd/llvm-obfuscator/plugins/linux-x86_64/upx \
    linux-x86_64/upx
```

This will:
- Download the existing tarball from GCP
- Add your UPX binary to it
- Upload the updated tarball back to GCP
- Make it available for CI/CD builds

## Step 2: Verify Upload

Check that the binary was added:

```bash
./scripts/gcp-binary-manager.sh list | grep upx
```

You should see:
```
linux-x86_64/upx
```

## Step 3: CI/CD Integration

The workflow (`.github/workflows/dockerhub-deploy.yml`) has been updated to:

1. **Download UPX from GCP** - The `download-binaries` job downloads the tarball which now includes UPX
2. **Make it executable** - Added `chmod +x` for the UPX binary
3. **Include in Docker build** - The Dockerfile copies it to `/usr/local/llvm-obfuscator/bin/upx`

## Step 4: Docker Image

The Dockerfile (`cmd/llvm-obfuscator/Dockerfile.backend`) has been updated to:

1. **Copy UPX binary** - Copies from `plugins/linux-x86_64/upx` to `/usr/local/llvm-obfuscator/bin/upx`
2. **Fallback to system UPX** - If custom UPX is not found, installs `upx-ucl` from apt
3. **Auto-detect in code** - The `UPXPacker` class automatically detects the Docker image UPX at `/usr/local/llvm-obfuscator/bin/upx`

## How It Works

### Priority Order

The `UPXPacker` resolves the UPX binary in this order:

1. **Custom path from config** - If `custom_upx_path` is set in config
2. **Docker image default** - `/usr/local/llvm-obfuscator/bin/upx` (your custom UPX)
3. **System UPX** - Falls back to `upx` from PATH (system installation)

### In Docker Container

When running in the Docker container:

```python
# No config needed - automatically uses Docker image UPX
packer = UPXPacker()  # Will find /usr/local/llvm-obfuscator/bin/upx
```

### With Custom Path

You can still override with a custom path:

```python
# Use a different UPX binary
packer = UPXPacker(custom_upx_path=Path("/custom/path/upx"))
```

## Testing Locally

Before pushing to Docker Hub, test locally:

```bash
# 1. Make sure UPX is in plugins directory
ls -lh cmd/llvm-obfuscator/plugins/linux-x86_64/upx

# 2. Build Docker image locally
cd cmd/llvm-obfuscator
docker build -f Dockerfile.backend -t llvm-obfuscator-backend:test .

# 3. Test that UPX is in the image
docker run --rm llvm-obfuscator-backend:test /usr/local/llvm-obfuscator/bin/upx --version
```

## Updating the UPX Binary

If you rebuild your custom UPX and want to update it:

```bash
# 1. Copy new binary
cp /home/dhruv/Documents/upx/build/upx cmd/llvm-obfuscator/plugins/linux-x86_64/upx
chmod +x cmd/llvm-obfuscator/plugins/linux-x86_64/upx

# 2. Upload to GCP
./scripts/gcp-binary-manager.sh add \
    cmd/llvm-obfuscator/plugins/linux-x86_64/upx \
    linux-x86_64/upx

# 3. Next CI/CD build will use the new binary
```

## Troubleshooting

### UPX not found in Docker image

Check the build logs:
```bash
# Look for this in CI/CD logs
✅ Custom UPX binary found
```

If you see "Custom UPX not found", verify:
1. Binary was uploaded to GCP: `./scripts/gcp-binary-manager.sh list | grep upx`
2. Binary is in the tarball: Check GCP bucket `gs://llvmbins/llvm-obfuscator-binaries.tar.gz`

### Fallback to system UPX

If custom UPX is not available, the Dockerfile will install `upx-ucl` from apt as a fallback. This is fine for basic functionality, but won't have your custom modifications.

## Summary

✅ **Upload once** - Use `gcp-binary-manager.sh` to upload UPX to GCP  
✅ **Automatic in CI/CD** - Workflow downloads it automatically  
✅ **Auto-detected in Docker** - Code finds it at `/usr/local/llvm-obfuscator/bin/upx`  
✅ **Fallback available** - System UPX if custom not found  

Your custom UPX is now part of the Docker image! 🎉


