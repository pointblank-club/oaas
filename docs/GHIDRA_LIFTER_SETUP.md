# Ghidra Lifter Setup & Configuration

## Overview

The Ghidra Lifter is Stage 1 of the Binary Obfuscation Pipeline. It extracts Control Flow Graphs (CFG) from Windows PE binaries using Ghidra 11.2.1 headless mode.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GHIDRA LIFTER SERVICE                    │
├─────────────────────────────────────────────────────────────┤
│  Container: oass-ghidra-lifter-1                            │
│  Port: 5000 (internal) → 5001 (host)                        │
│  Image: pb-ghidra-lifter:latest                             │
│                                                              │
│  Components:                                                 │
│  ├── Ghidra 11.2.1 (headless mode)                          │
│  ├── Java 21 (OpenJDK)                                      │
│  ├── Flask API (lifter_service.py)                          │
│  └── Export Script (export_cfg.py)                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Location | Purpose |
|------|----------|---------|
| Dockerfile | `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/Dockerfile` | Container build |
| lifter_service.py | `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/lifter_script/lifter_service.py` | Flask API |
| export_cfg.py | `binary_obfuscation_pipeline/mcsema_impl/ghidra_lifter/lifter_script/export_cfg.py` | Ghidra script |

## Critical Requirements

### Java 21 (NOT Java 17!)

Ghidra 11.2.1 requires **Java 21**. Using Java 17 results in:
```
java.lang.UnsupportedClassVersionError: ghidra/GhidraClassLoader has been
compiled by a more recent version of the Java Runtime (class file version 65.0),
this version of the Java Runtime only recognizes class file versions up to 61.0
```

**Class file versions:**
- Java 17 = class file 61.0
- Java 21 = class file 65.0

### JAVA_HOME Configuration

**CRITICAL:** The `JAVA_HOME` must be **hardcoded** in `lifter_service.py`:

```python
# CORRECT - Hardcoded path
JAVA_HOME = '/usr/lib/jvm/java-21-openjdk'

# WRONG - May pick up Java 17 from container env
JAVA_HOME = os.getenv('JAVA_HOME', '/usr/lib/jvm/java-21-openjdk')
```

The container's environment may have `JAVA_HOME` set to Java 17, so `os.getenv()` will return the wrong value.

## Dockerfile Configuration

```dockerfile
# Install Java 21 (NOT Java 17!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    ...

# Set JAVA_HOME dynamically for multi-arch support
RUN JAVA_DIR=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "export JAVA_HOME=$JAVA_DIR" >> /etc/profile.d/java.sh && \
    ln -sf $JAVA_DIR /usr/lib/jvm/java-21-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk

# Patch Ghidra launch.sh to use environment JAVA_HOME
sed -i '136s|.*|if [ -n "${JAVA_HOME}" ] \&\& [ -x "${JAVA_HOME}/bin/java" ]; then : ; else JAVA_HOME="$(java -cp "${LS_CPATH}" LaunchSupport "${INSTALL_DIR}" ${JAVA_TYPE_ARG} -save)"; fi|' \
    /opt/ghidra/ghidra_11.2.1_PUBLIC/support/launch.sh
```

## API Endpoints

### POST /lift
Upload binary file directly.

```bash
curl -X POST http://localhost:5001/lift \
  -F "file=@binary.exe" \
  -F "output_cfg_path=/app/reports/output.cfg"
```

### POST /lift/file
Reference binary by path (for volume-mounted files).

```bash
curl -X POST http://localhost:5001/lift/file \
  -H "Content-Type: application/json" \
  -d '{"binary_path": "/app/binaries/input.exe", "output_cfg_path": "/app/reports/output.cfg"}'
```

### GET /health
Health check.

```bash
curl http://localhost:5001/health
# Returns: {"status": "healthy", "service": "ghidra-lifter"}
```

## CFG Output Format

```json
{
  "binary_name": "input.exe",
  "architecture": "x86_64",
  "functions": [
    {
      "name": "main",
      "address": "0x00401000",
      "size": 256,
      "basic_blocks": [
        {
          "address": "0x00401000",
          "size": 32,
          "instructions": [
            {"address": "0x00401000", "mnemonic": "push", "operands": "rbp"},
            {"address": "0x00401001", "mnemonic": "mov", "operands": "rbp, rsp"}
          ]
        }
      ],
      "edges": [
        {"from": "0x00401000", "to": "0x00401020", "type": "flow"},
        {"from": "0x00401020", "to": "0x00401040", "type": "branch"}
      ]
    }
  ]
}
```

## Ghidra API Compatibility Notes

### InstructionDB has no `isBranch()` method

**Wrong:**
```python
is_branch = instr.isBranch()  # AttributeError!
```

**Correct:**
```python
flow_type = instr.getFlowType()
is_branch = flow_type.isJump() or flow_type.isConditional()
```

### AddressSet has no `getLength()` method

**Wrong:**
```python
size = func.getBody().getLength()  # AttributeError!
```

**Correct:**
```python
size = func.getBody().getNumAddresses()
```

## Troubleshooting

### "UnsupportedClassVersionError: class file version 65.0"
**Cause:** Java 17 is being used instead of Java 21.
**Fix:** Ensure Java 21 is installed and JAVA_HOME is set correctly.

```bash
docker exec oass-ghidra-lifter-1 java -version
# Should show: openjdk version "21.x.x"
```

### "Unable to prompt user for JDK path"
**Cause:** Ghidra's launch.sh can't find Java.
**Fix:** Patch launch.sh line 136 (done in Dockerfile).

### Flask not picking up code changes
**Cause:** Flask doesn't hot-reload.
**Fix:** Rebuild and recreate container:
```bash
docker compose build ghidra-lifter && docker compose up -d ghidra-lifter
```

## Build & Deploy

### Build Image
```bash
docker compose build ghidra-lifter
```

### Start Container
```bash
docker compose up -d ghidra-lifter
```

### Rebuild & Restart
```bash
docker compose build ghidra-lifter && docker compose up -d ghidra-lifter
```

### View Logs
```bash
docker logs oass-ghidra-lifter-1 --tail 50
```

### Verify Ghidra Works
```bash
docker exec oass-ghidra-lifter-1 bash -c '
  JAVA_HOME=/usr/lib/jvm/java-21-openjdk \
  /opt/ghidra/ghidra_11.2.1_PUBLIC/support/analyzeHeadless -version
'
```

## GCP Build VM

For building McSema and other heavy binaries, use the GCP VM:

```bash
# SSH into the llvm build VM
gcloud compute ssh llvm --zone=<zone> --project=unified-coyote-478817-r3

# Or direct SSH (if you have the IP)
ssh user@<VM_IP>
```

**Note:** The service account `github-actions-binaries@unified-coyote-478817-r3.iam.gserviceaccount.com` has storage permissions but NOT compute permissions. For VM access, use your personal GCP credentials or add compute permissions to the service account.
