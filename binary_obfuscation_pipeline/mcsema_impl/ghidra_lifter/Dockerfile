# Ghidra Lifter Container for CFG Export
# Produces McSema-compatible control flow graphs from Windows PE binaries

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
# NOTE: Ghidra 11.2.1 requires Java 21, not Java 17!
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    python3 \
    python3-pip \
    wget \
    unzip \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME dynamically based on architecture (arm64 or amd64)
RUN JAVA_DIR=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "export JAVA_HOME=$JAVA_DIR" >> /etc/profile.d/java.sh && \
    ln -sf $JAVA_DIR /usr/lib/jvm/java-21-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk

# Create app directory
WORKDIR /app

# Download and install Ghidra (latest stable)
# Ghidra is downloaded from official release repository
# We use headless mode only—no GUI components needed
# Using Ghidra 11.2.1 for stability (well-tested version)
RUN mkdir -p /opt/ghidra && \
    cd /tmp && \
    wget -q https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.2.1_build/ghidra_11.2.1_PUBLIC_20241105.zip && \
    unzip -q ghidra_11.2.1_PUBLIC_20241105.zip -d /opt/ghidra && \
    rm ghidra_11.2.1_PUBLIC_20241105.zip && \
    chmod +x /opt/ghidra/ghidra_11.2.1_PUBLIC/support/analyzeHeadless && \
    # Configure Ghidra to use our Java 21 installation
    sed -i 's|^JAVA_HOME_OVERRIDE=.*|JAVA_HOME_OVERRIDE=/usr/lib/jvm/java-21-openjdk|' \
        /opt/ghidra/ghidra_11.2.1_PUBLIC/support/launch.properties && \
    # Patch launch.sh to properly use JAVA_HOME from environment (line 136)
    sed -i '136s|.*|if [ -n "${JAVA_HOME}" ] \&\& [ -x "${JAVA_HOME}/bin/java" ]; then : ; else JAVA_HOME="$(java -cp "${LS_CPATH}" LaunchSupport "${INSTALL_DIR}" ${JAVA_TYPE_ARG} -save)"; fi|' \
        /opt/ghidra/ghidra_11.2.1_PUBLIC/support/launch.sh

# Install Python dependencies for the lifter script
RUN pip install --no-cache-dir \
    flask \
    requests \
    pyyaml

# Copy the Ghidra lifter scripts
COPY lifter_script/ /app/lifter_script/

# Create reports directory for output
RUN mkdir -p /app/reports /app/binaries /app/binary_jobs

# Set environment variables
ENV GHIDRA_HOME=/opt/ghidra/ghidra_11.2.1_PUBLIC
ENV PATH="${GHIDRA_HOME}/support:${PATH}"

# Health check: verify Ghidra can start
HEALTHCHECK CMD ${GHIDRA_HOME}/support/analyzeHeadless -version 2>/dev/null || exit 1

EXPOSE 5000

# Run the lifter service
CMD ["python3", "/app/lifter_script/lifter_service.py"]
