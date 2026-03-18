import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import config
# from app.api.routes import chat, document

# ---------------------------------------------------------
# LangSmith setup
# ---------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT

# ---------------------------------------------------------
# FastAPI app initialization
# ---------------------------------------------------------
app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
)

# ---------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Create uploads directory if it doesn't exist
# ---------------------------------------------------------
os.makedirs(config.UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "app": config.APP_TITLE,
        "version": config.APP_VERSION,
    }

# NOTE: Route imports will be added here as each route is built
# from app.api.routes import chat, document
# app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
# app.include_router(document.router, prefix="/api/v1", tags=["Document"])