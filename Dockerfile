# Matches the Python version pinned for bare-metal dev (see requirements.txt):
# langchain 0.2.x pins numpy<2.0, which has no prebuilt wheel for Python
# 3.13+, so 3.12 is the newest interpreter with full wheel coverage for
# this stack. "slim" (not "alpine") because alpine's musl libc breaks
# prebuilt manylinux wheels for compiled deps like chroma-hnswlib/numpy,
# which would force compiling them from source inside the image - slower
# builds and needing a full C toolchain for zero real benefit here.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies BEFORE copying the rest of the app. Docker caches
# each layer keyed on its inputs - as long as requirements.txt is
# unchanged, `docker build` reuses this layer instead of re-running pip
# install, so editing rag_api.py doesn't trigger a multi-minute reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user. The app has no reason to run as root, and a
# container escape from a root process is a strictly worse day than one
# from an unprivileged process - a cheap, standard hardening step.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Reuses the /health/ endpoint built in Phase 2 specifically for this kind
# of check. A one-line Python call (httpx is already a dependency) instead
# of `curl` avoids an extra apt-get install just for the healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health/', timeout=3).status_code==200 else 1)"

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
