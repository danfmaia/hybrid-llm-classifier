from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from ..models.classifier import HybridClassifier, ClassificationResult
from ..auth.jwt import get_current_user
from ..config import get_settings

router = APIRouter(prefix="/api/v1/classify", tags=["classification"])


class ClassificationRequest(BaseModel):
    """Request model for document classification."""
    text: str = Field(..., min_length=1, max_length=50000)
    metadata: Optional[dict] = None


@router.post("/", response_model=ClassificationResult)
async def classify_document(
    request: ClassificationRequest,
    classifier: HybridClassifier = Depends(get_settings().get_classifier),
    current_user: dict = Depends(get_current_user)
) -> ClassificationResult:
    """
    Classify a legal document using the hybrid classification system.

    Args:
        request: Classification request containing document text
        classifier: Injected classifier instance
        current_user: JWT authenticated user information

    Returns:
        ClassificationResult with category and confidence scores

    Raises:
        HTTPException: If classification fails or input is invalid
    """
    try:
        result = await classifier.classify(request.text)
        return result
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Classification not yet implemented"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/batch", response_model=List[ClassificationResult])
async def classify_documents(
    requests: List[ClassificationRequest],
    classifier: HybridClassifier = Depends(get_settings().get_classifier),
    current_user: dict = Depends(get_current_user)
) -> List[ClassificationResult]:
    """Batch classify multiple documents."""
    results = []
    for request in requests:
        try:
            result = await classifier.classify(request.text)
            results.append(result)
        except NotImplementedError:
            results.append(ClassificationResult(
                category="error",
                confidence=0.0,
                error="Classification not yet implemented"
            ))
        except Exception as e:
            results.append(ClassificationResult(
                category="error",
                confidence=0.0,
                error=str(e)
            ))
    return results
