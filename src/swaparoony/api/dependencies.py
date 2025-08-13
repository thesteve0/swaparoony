from functools import lru_cache
from ..services.face_swap_service import FaceSwapService

# Global service instance
_face_swap_service = None


def get_face_swap_service() -> FaceSwapService:
    """Dependency injection for face swap service"""
    global _face_swap_service
    if _face_swap_service is None:
        _face_swap_service = FaceSwapService()
        _face_swap_service.initialize_models()
    return _face_swap_service
