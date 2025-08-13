from fastapi import UploadFile
from pathlib import Path
from ..core.config import settings
from ..core.exceptions import InvalidImageError


async def validate_image_file(file: UploadFile) -> bytes:
    """Validate and read uploaded image file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise InvalidImageError(
            f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}"
        )

    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_file_size:
        raise InvalidImageError(
            f"File too large. Max size: {settings.max_file_size // (1024*1024)}MB"
        )

    return contents
