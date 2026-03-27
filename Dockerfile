# ---------------------------------------------------------
# LexAI — Agentic Law Assistant
# ---------------------------------------------------------
    FROM python:3.11-slim

    RUN pip install --no-cache-dir uv

    WORKDIR /app
    COPY requirements.txt .
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv pip install --system -r requirements.txt

    COPY . .
    
    # Create necessary directories
    RUN mkdir -p uploads faiss_index
    
    # Expose FastAPI port
    EXPOSE 8000
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
        CMD curl -f http://localhost:8000/health || exit 1
    
    # Start FastAPI server
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]