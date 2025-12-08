# Ghidra Lifter Container for CFG Export
# Produces McSema-compatible control flow graphs from Windows PE binaries

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    python3 \
    python3-pip \
    wget \
    unzip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Download and install Ghidra (latest stable)
# Ghidra is downloaded from official release repository
# We use headless mode only—no GUI components needed
RUN mkdir -p /opt/ghidra && \
    cd /tmp && \
    wget -q https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.1.1_build/ghidra_11.1.1_PUBLIC_20240723.zip && \
    unzip -q ghidra_11.1.1_PUBLIC_20240723.zip -d /opt/ghidra && \
    rm ghidra_11.1.1_PUBLIC_20240723.zip && \
    chmod +x /opt/ghidra/ghidra_11.1.1_PUBLIC/support/analyzeHeadless

# Install Python dependencies for the lifter script
RUN pip install --no-cache-dir \
    requests \
    pyyaml

# Copy the Ghidra lifter scripts
COPY lifter_script/ /app/lifter_script/

# Create reports directory for output
RUN mkdir -p /app/reports /app/binaries

# Set environment variables
ENV GHIDRA_HOME=/opt/ghidra/ghidra_11.1.1_PUBLIC
ENV PATH="${GHIDRA_HOME}/support:${PATH}"

# Health check: verify Ghidra can start
HEALTHCHECK CMD ${GHIDRA_HOME}/support/analyzeHeadless -version 2>/dev/null || exit 1

EXPOSE 5000

# Run the lifter service
CMD ["python3", "/app/lifter_script/lifter_service.py"]
