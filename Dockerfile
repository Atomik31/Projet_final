# ============================================================
# Dockerfile — Wind Turbine Maintenance Dashboard
# Compatible HuggingFace Spaces (port 7860, user uid=1000)
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user required by HuggingFace Spaces
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Expose Streamlit port (HuggingFace default)
EXPOSE 7860

# Streamlit config is in streamlit/.streamlit/config.toml
# but we override port & address here to be explicit
CMD ["streamlit", "run", "streamlit/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]
