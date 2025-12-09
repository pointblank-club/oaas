# McSema Lift Service
# IMPORTANT: Using LLVM 9 - LLVM 11 has CallSite bug that causes segfaults
# Use: docker build --platform linux/amd64 -t mcsema-lift .

FROM --platform=linux/amd64 trailofbits/mcsema:llvm9-ubuntu20.04-amd64

LABEL maintainer="oaas-team"
LABEL description="McSema Lift Service - Binary to LLVM IR lifter (LLVM 9)"

# Install Flask for the service API (pip3 already available in base image)
RUN pip3 install --no-cache-dir flask requests

# Create app directories
RUN mkdir -p /app /app/cfg /app/output

WORKDIR /app

# Copy the service script
COPY lift_service.py /app/lift_service.py

# Expose the service port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Clear base image ENTRYPOINT and set our service as CMD
ENTRYPOINT []
CMD ["python3", "/app/lift_service.py"]
