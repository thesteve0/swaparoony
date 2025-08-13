from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import List

from ...services.face_swap_service import FaceSwapService
from ...models.schemas import FaceSwapResponse, SwappedImage, ErrorResponse
from ...utils.image_utils import validate_image_file
from ...api.dependencies import get_face_swap_service
from ...core.exceptions import (
    NoFaceDetectedError,
    InsufficientFacesError,
    InvalidImageError,
    FaceSwapError,
)

router = APIRouter()


@router.post("/swap", response_model=FaceSwapResponse)
async def swap_faces(
    image: UploadFile = File(..., description="Source image with face to swap"),
    source_face_id: int = Form(
        1, ge=1, description="Face position in source image (starting at 1)"
    ),
    destination_face_id: int = Form(
        1, ge=1, description="Face position in destination images (starting at 1)"
    ),
    service: FaceSwapService = Depends(get_face_swap_service),
):
    """
    Swap face from uploaded image onto all configured destination images
    """
    try:
        # Validate and read image
        image_data = await validate_image_file(image)

        # Process face swap
        results, faces_detected = service.process_face_swap_request(
            source_image_data=image_data,
            source_face_id=source_face_id,
            dest_face_id=destination_face_id,
        )

        # Build response
        swapped_images = [
            SwappedImage(image_data=base64_data, destination_name=filename)
            for base64_data, filename in results
        ]

        return FaceSwapResponse(
            success=True,
            message=f"Successfully swapped face onto {len(swapped_images)} images",
            swapped_images=swapped_images,
            faces_detected_in_source=faces_detected,
        )

    except NoFaceDetectedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InsufficientFacesError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FaceSwapError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/health")
async def health_check(service: FaceSwapService = Depends(get_face_swap_service)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": service._initialized,
        "destination_images_count": len(service.destination_images),
    }
