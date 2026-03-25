import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.chat_schema import DocumentUploadResponse
from app.rag.document_loader import load_pdf, extract_doc_date
from app.rag.chunker import chunk_documents
from app.rag.embedder import create_vector_store
from app.core.config import config

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Accepts a PDF upload, validates it, chunks and indexes
    it into a Qdrant collection for the session.
    """

    if file.content_type not in config.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is "
                   f"{config.MAX_FILE_SIZE_MB}MB.",
        )

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_name = f"{session_id}_{uuid.uuid4().hex[:8]}.pdf"
    file_path = os.path.join(config.UPLOAD_DIR, file_name)

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        documents = load_pdf(file_path)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF.",
            )

        doc_date = extract_doc_date(file_path)
        chunks = chunk_documents(documents)

        # Qdrant: pass session_id directly
        create_vector_store(chunks, session_id)

        return DocumentUploadResponse(
            success=True,
            message=(
                f"Document processed successfully. "
                f"{len(documents)} pages loaded, "
                f"{len(chunks)} chunks indexed into Qdrant."
            ),
            doc_path=file_path,
            doc_date=doc_date,
        )

    except HTTPException:
        raise

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}",
        )