from pydantic import BaseModel, Field
from typing import List, Optional
import base64


class SwappedImage(BaseModel):
    image_data: str = Field(description="Base64 encoded image")
    destination_name: str = Field(description="Name of the destination image used")


class FaceSwapResponse(BaseModel):
    success: bool
    message: str
    swapped_images: List[SwappedImage] = []
    faces_detected_in_source: int = 0


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
