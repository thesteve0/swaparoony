import cv2
import numpy as np
import base64
from typing import List, Tuple
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

from ..core.config import settings
from ..core.exceptions import (
    NoFaceDetectedError,
    InsufficientFacesError,
    InvalidImageError,
    ModelLoadError,
)


class FaceSwapService:
    def __init__(self):
        self.app = None
        self.swapper = None
        self.destination_images = []  # List of (image_array, filename) tuples
        self._initialized = False

    def initialize_models(self):
        """Initialize face analysis and swapper models, preload destination images"""
        try:
            self.app = FaceAnalysis(name=settings.face_analysis_name, allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=settings.ctx_id, det_size=settings.det_size)

            self.swapper = insightface.model_zoo.get_model(
                settings.model_path, download=False, download_zip=False
            )

            # Preload destination images
            self._load_destination_images()

            self._initialized = True
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize models: {str(e)}")

    def _load_destination_images(self):
        """Load all destination images into memory"""
        self.destination_images = []

        for dest_path in settings.destination_images:
            path = Path(dest_path)
            if not path.exists():
                print(f"Warning: Destination image not found: {dest_path}")
                continue

            image = cv2.imread(str(path))
            if image is None:
                print(f"Warning: Could not load image: {dest_path}")
                continue

            self.destination_images.append((image, path.name))

        if not self.destination_images:
            raise ModelLoadError("No destination images could be loaded")

        print(f"Loaded {len(self.destination_images)} destination images into memory")

    def _ensure_initialized(self):
        if not self._initialized:
            raise ModelLoadError(
                "Models not initialized. Call initialize_models() first."
            )

    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image from bytes to numpy array"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise InvalidImageError("Could not decode image")
            return image
        except Exception as e:
            raise InvalidImageError(f"Invalid image format: {str(e)}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy array to base64 string"""
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def _get_faces(self, image: np.ndarray) -> List:
        """Get sorted faces from image"""
        faces = self.app.get(image)
        return sorted(faces, key=lambda x: x.bbox[0])

    def _validate_face_index(self, faces: List, face_index: int, image_type: str):
        """Validate that face index exists in detected faces"""
        if not faces:
            raise NoFaceDetectedError(f"No faces detected in {image_type} image")

        if len(faces) < face_index or face_index < 1:
            raise InsufficientFacesError(
                f"{image_type} image contains {len(faces)} faces, "
                f"but requested face {face_index}"
            )

    def swap_face_on_image(
        self,
        source_image: np.ndarray,
        destination_image: np.ndarray,
        source_face_id: int = 1,
        dest_face_id: int = 1,
    ) -> np.ndarray:
        """Swap face from source onto destination image"""
        self._ensure_initialized()

        # Get faces from both images
        source_faces = self._get_faces(source_image)
        dest_faces = self._get_faces(destination_image)

        # Validate face indices
        self._validate_face_index(source_faces, source_face_id, "source")
        self._validate_face_index(dest_faces, dest_face_id, "destination")

        # Get specific faces
        source_face = source_faces[source_face_id - 1]
        dest_face = dest_faces[dest_face_id - 1]

        # Perform face swap
        result = self.swapper.get(
            destination_image, dest_face, source_face, paste_back=True
        )

        return result

    def process_face_swap_request(
        self, source_image_data: bytes, source_face_id: int = 1, dest_face_id: int = 1
    ) -> Tuple[List[Tuple[str, str]], int]:
        """
        Process face swap for all preloaded destination images
        Returns: (list_of_(base64_image, filename)_tuples, faces_detected_in_source)
        """
        self._ensure_initialized()

        # Decode source image
        source_image = self._decode_image(source_image_data)

        # Get source faces for validation and count
        source_faces = self._get_faces(source_image)
        self._validate_face_index(source_faces, source_face_id, "source")

        results = []

        for dest_image, filename in self.destination_images:
            try:
                # Perform face swap on preloaded image
                swapped = self.swap_face_on_image(
                    source_image, dest_image, source_face_id, dest_face_id
                )

                # Encode to base64
                encoded = self._encode_image(swapped)
                results.append((encoded, filename))

            except Exception:
                # Skip this destination if swap fails
                continue

        return results, len(source_faces)
