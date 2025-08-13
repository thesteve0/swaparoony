class FaceSwapError(Exception):
    """Base exception for face swap operations"""

    pass


class NoFaceDetectedError(FaceSwapError):
    """Raised when no faces are detected in an image"""

    pass


class InsufficientFacesError(FaceSwapError):
    """Raised when requested face index exceeds detected faces"""

    pass


class InvalidImageError(FaceSwapError):
    """Raised when image format is invalid or corrupted"""

    pass


class ModelLoadError(FaceSwapError):
    """Raised when face swap models fail to load"""

    pass
