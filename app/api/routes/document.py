import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.chat_schema import DocumentUploadResponse
from app.rag.document_loader import load_pdf, extract_doc_date
from app.rag.chunker import chunk_documents
from app.rag.embedder import create_vector_store, save_vector_store
from app.core.config import config

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Accepts a PDF upload from the frontend.
    Validates, chunks, embeds, and stores it
    in a FAISS vector store under the session_id.

    Args:
        session_id: Unique session identifier.
        file: Uploaded PDF file.

    Returns:
        DocumentUploadResponse with success status and doc metadata.
    """

    # Validate file type
    if file.content_type not in config.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > config.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is "
                   f"{config.MAX_FILE_SIZE_MB}MB.",
        )

    # Save file to uploads directory
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_name = f"{session_id}_{uuid.uuid4().hex[:8]}.pdf"
    file_path = os.path.join(config.UPLOAD_DIR, file_name)

    with open(file_path, "wb") as f:
        f.write(contents)

    # Process document
    try:
        documents = load_pdf(file_path)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. "
                       "File may be scanned or corrupted.",
            )

        # Extract document date
        doc_date = extract_doc_date(file_path)

        # Chunk and embed
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)
        save_vector_store(vector_store, session_id)

        return DocumentUploadResponse(
            success=True,
            message=f"Document processed successfully. "
                    f"{len(chunks)} chunks indexed.",
            doc_path=file_path,
            doc_date=doc_date,
        )

    except HTTPException:
        raise

    except Exception as e:
        # Clean up file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}",
        )