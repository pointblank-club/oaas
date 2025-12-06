# OAAS - LLVM Obfuscator Project

## Project Overview

1. OLLVM code is in `../llvm-project` under `ollvm-integration` branch → LLVM 22
2. Deployment is at https://oaas.pointblank.club
3. We always test by deploying to server using Docker and testing with Playwright end-to-end
4. Our original repo ran out of CI credits, now we push to a fork : https://github.com/slashexx/oaas

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
