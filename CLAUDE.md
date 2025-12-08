# OAAS - LLVM Obfuscator Project

## Project Overview

1. OLLVM code is in `../llvm-project` under `ollvm-integration` branch → LLVM 22
2. Deployment is at https://oaas.pointblank.club
3. NEVER EVER CO AUTHOR CLAUDE OR MENTION CLAUDE IN ANY PR OR COMMITS
4. ALWAYS WRITE .md FILES TO /docs DIR AND ONLY CREATE NEW .md file IF IT CANNOT BE ADDED / MERGED IN AN EXISTING DOC 
5. We push and raise PR's to https://github.com/slashexx/oaas


## Production Server

- **Host**: `root@69.62.77.147`
- **Deployment Directory**: `/home/devopswale/oaas/`
- **URL**: https://oaas.pointblank.club

## Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Server                         │
├─────────────────────────────────────────────────────────────┤
│  llvm-obfuscator-frontend (port 4666)                       │
│    - Image: akashsingh04/llvm-obfuscator-frontend:latest    │
│    - Network: oaas_obfuscator-network                       │
│                                                              │
│  llvm-obfuscator-backend (port 8000)                        │
│    - Image: akashsingh04/llvm-obfuscator-backend:patched    │
│    - Network: oaas_obfuscator-network                       │
│    - Volumes: ./reports, ./logs                             │
└─────────────────────────────────────────────────────────────┘
```

## Manual Deployment Guide

### 1. Deploy Python Code Changes (Backend)

For changes to Python files (`api/`, `core/`, `cli/`):

```bash
# Copy the changed file(s) to server
scp cmd/llvm-obfuscator/api/server.py root@69.62.77.147:/tmp/
scp cmd/llvm-obfuscator/core/obfuscator.py root@69.62.77.147:/tmp/

# Update container and restart
ssh root@69.62.77.147 "
  docker cp /tmp/server.py llvm-obfuscator-backend:/app/api/server.py
  docker cp /tmp/obfuscator.py llvm-obfuscator-backend:/app/core/obfuscator.py
  docker restart llvm-obfuscator-backend
"

# Verify it's running
ssh root@69.62.77.147 "docker logs llvm-obfuscator-backend --tail 10"
```

### 2. Deploy Frontend Changes

For changes to `frontend/src/`:

```bash
# Build frontend image on server (from local source)
tar -czf /tmp/frontend-src.tar.gz -C cmd/llvm-obfuscator/frontend --exclude='node_modules' --exclude='dist' .
scp /tmp/frontend-src.tar.gz root@69.62.77.147:/tmp/

# On server: extract, build, and restart
ssh root@69.62.77.147 "
  rm -rf /tmp/frontend-build && mkdir -p /tmp/frontend-build
  cd /tmp/frontend-build && tar -xzf /tmp/frontend-src.tar.gz
  docker build -f Dockerfile.frontend -t akashsingh04/llvm-obfuscator-frontend:latest .
  docker stop llvm-obfuscator-frontend && docker rm llvm-obfuscator-frontend
  docker run -d --name llvm-obfuscator-frontend --network oaas_obfuscator-network -p 4666:4666 --restart unless-stopped akashsingh04/llvm-obfuscator-frontend:latest
"
```

### 3. Deploy LLVM Plugin Binary

For updated `LLVMObfuscationPlugin.so`:

```bash
# Copy new plugin to server
scp /path/to/LLVMObfuscationPlugin.so root@69.62.77.147:/tmp/

# Update both locations in container
ssh root@69.62.77.147 "
  docker cp /tmp/LLVMObfuscationPlugin.so llvm-obfuscator-backend:/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so
  docker cp /tmp/LLVMObfuscationPlugin.so llvm-obfuscator-backend:/app/plugins/linux-x86_64/LLVMObfuscationPlugin.so
  docker exec llvm-obfuscator-backend chmod +x /usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so
  docker exec llvm-obfuscator-backend chmod +x /app/plugins/linux-x86_64/LLVMObfuscationPlugin.so
  docker restart llvm-obfuscator-backend
"

# Verify plugin updated
ssh root@69.62.77.147 "docker exec llvm-obfuscator-backend ls -lh /usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so"
```

### 4. Deploy LLVM Binaries (clang, opt)

For updated `clang` or `opt` binaries:

```bash
# Copy binary to server
scp /path/to/clang root@69.62.77.147:/tmp/
scp /path/to/opt root@69.62.77.147:/tmp/

# Update in container (clang goes to clang.real, there's a wrapper)
ssh root@69.62.77.147 "
  docker cp /tmp/clang llvm-obfuscator-backend:/usr/local/llvm-obfuscator/bin/clang.real
  docker cp /tmp/opt llvm-obfuscator-backend:/usr/local/llvm-obfuscator/bin/opt
  docker exec llvm-obfuscator-backend chmod +x /usr/local/llvm-obfuscator/bin/clang.real
  docker exec llvm-obfuscator-backend chmod +x /usr/local/llvm-obfuscator/bin/opt
  docker restart llvm-obfuscator-backend
"
```

## Key File Locations in Container

| File Type | Container Path |
|-----------|----------------|
| Python API | `/app/api/server.py` |
| Python Core | `/app/core/*.py` |
| LLVM Plugin | `/usr/local/llvm-obfuscator/lib/LLVMObfuscationPlugin.so` |
| LLVM Plugin (backup) | `/app/plugins/linux-x86_64/LLVMObfuscationPlugin.so` |
| Clang binary | `/usr/local/llvm-obfuscator/bin/clang.real` |
| Clang wrapper | `/usr/local/llvm-obfuscator/bin/clang` (symlink) |
| Opt binary | `/usr/local/llvm-obfuscator/bin/opt` |
| Wrapper scripts | `/usr/local/llvm-obfuscator/bin/clang-obfuscate`, `clang++-obfuscate` |

## Useful Commands

```bash
# Check container status
ssh root@69.62.77.147 "docker ps | grep obfuscator"

# View backend logs
ssh root@69.62.77.147 "docker logs llvm-obfuscator-backend --tail 50"

# View frontend logs
ssh root@69.62.77.147 "docker logs llvm-obfuscator-frontend --tail 50"

# Restart backend
ssh root@69.62.77.147 "docker restart llvm-obfuscator-backend"

# Restart frontend
ssh root@69.62.77.147 "docker restart llvm-obfuscator-frontend"

# Check health
ssh root@69.62.77.147 "curl -s http://localhost:8000/api/health"

# Execute command in backend container
ssh root@69.62.77.147 "docker exec llvm-obfuscator-backend <command>"

# Check LLVM binaries
ssh root@69.62.77.147 "docker exec llvm-obfuscator-backend ls -lh /usr/local/llvm-obfuscator/bin/"
```

## CI/CD Pipeline (GitHub Actions)

The automated pipeline is in `.github/workflows/`:

1. **dockerhub-deploy.yml** - Builds and pushes images to Docker Hub (requires LLVM binaries in GitHub release)
2. **deploy.yml** - Deploys to server by pulling from Docker Hub

**Note**: CI/CD currently fails due to Git LFS quota exceeded. Use manual deployment above.

## Testing with Playwright

After deployment, test end-to-end:

```bash
# Navigate to site, select demo, enable all layers, run obfuscation
# Use Playwright MCP tools or manual testing at https://oaas.pointblank.club
```

## Network Configuration

- Frontend exposed on port `4666` (mapped to container port 4666)
- Backend internal on port `8000` (accessed via Docker network)
- Both containers on `oaas_obfuscator-network` bridge network

---

## GCP Binary Management

All LLVM binaries are stored in a **single tarball** (`llvm-obfuscator-binaries.tar.gz`) in GCP Cloud Storage.

### Quick Reference

```bash
# List all binaries in the tarball
./scripts/gcp-binary-manager.sh list

# Add/update a binary
./scripts/gcp-binary-manager.sh add /path/to/binary linux-x86_64/binary-name

# Download binaries for local development
./scripts/gcp-binary-manager.sh download
```

### Prerequisites

1. **Google Cloud SDK** installed:
   ```bash
   brew install google-cloud-sdk  # macOS
   ```

2. **Authenticated with GCP**:
   ```bash
   gcloud auth login
   ```

### Tarball Structure

```
llvm-obfuscator-binaries.tar.gz
├── linux-x86_64/                    # Linux binaries
│   ├── clang                        # Clang compiler
│   ├── opt                          # LLVM optimizer
│   ├── mlir-opt                     # MLIR optimizer
│   ├── clangir                      # ClangIR compiler
│   ├── LLVMObfuscationPlugin.so     # Obfuscation plugin
│   ├── MLIRObfuscation.so           # MLIR obfuscation plugin
│   ├── libLLVM.so.22.0git           # LLVM shared library
│   └── lib/clang/22/include/        # Clang headers
├── macos-sdk/                       # macOS cross-compilation SDK
│   └── MacOSX15.4.sdk/
└── llvm-mingw/                      # Windows cross-compilation toolchain
```

### Adding/Updating a Binary

```bash
# Add a new clang binary
./scripts/gcp-binary-manager.sh add /path/to/new/clang linux-x86_64/clang

# Add a new plugin
./scripts/gcp-binary-manager.sh add ./MyNewPlugin.so linux-x86_64/MyNewPlugin.so

# Add a directory (e.g., clang headers)
./scripts/gcp-binary-manager.sh add ./include linux-x86_64/lib/clang/22/include
```

**What happens internally:**
1. Downloads existing tarball from GCP
2. Extracts to temp directory
3. Adds/overwrites your new binary at the specified path
4. Creates new tarball
5. Uploads back to GCP

### Removing a Binary

```bash
./scripts/gcp-binary-manager.sh remove linux-x86_64/old-binary
```

### Example: Updating the Obfuscation Plugin

```bash
# 1. Build your new plugin locally
cd ../llvm-project
ninja LLVMObfuscationPlugin

# 2. Add it to the tarball
cd /path/to/oaas
./scripts/gcp-binary-manager.sh add \
    ../llvm-project/build/lib/LLVMObfuscationPlugin.so \
    linux-x86_64/LLVMObfuscationPlugin.so

# 3. Verify it was added
./scripts/gcp-binary-manager.sh list | grep Plugin

# 4. Push to trigger CI (uses updated tarball automatically)
git push
```

### CI/CD Integration

The CI workflow (`dockerhub-deploy.yml`) automatically:
1. **Tries tarball first** (fast path, ~2 mins)
2. **Falls back to individual files** if tarball doesn't exist (slow path, ~20 mins)

### GCP Bucket

```
gs://llvmbins/
├── llvm-obfuscator-binaries.tar.gz  # PRIMARY: Single tarball
├── linux-x86_64/                     # LEGACY: Individual files (fallback)
├── macos-sdk-15.4-minimal.tar.gz    # LEGACY: Separate SDK
└── llvm-mingw-*.tar.xz              # LEGACY: Separate MinGW
```

### Troubleshooting

**"gsutil not found"**: `brew install google-cloud-sdk`

**"Not authenticated"**: `gcloud auth login`

**"Permission denied"**: Contact project owner for `Storage Object Admin` role on `llvmbins` bucket
