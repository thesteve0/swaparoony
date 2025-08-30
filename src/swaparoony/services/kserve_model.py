import logging
from typing import Dict, Any
import kserve
from kserve import ModelServer
import base64
from .face_swap_service import FaceSwapService
from ..core.exceptions import (
    ModelLoadError,
    NoFaceDetectedError,
    InsufficientFacesError,
    InvalidImageError,
    FaceSwapError,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KServeFaceSwapModel(kserve.Model):
    """
    Minimal KServe model for face swap functionality.
    This initial version just logs method calls to validate KServe integration.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        logger.info(f"Initializing KServeFaceSwapModel: {name}")
        self.ready = False
        self.face_swap_service = None

    def load(self):
        """Load the face swap models and initialize the service"""
        logger.info("load() method called - loading face swap models")

        try:
            # Initialize the actual FaceSwapService
            logger.info("Initializing FaceSwapService...")
            self.face_swap_service = FaceSwapService()
            self.face_swap_service.initialize_models()

            logger.info(
                f"Face swap models loaded successfully! {len(self.face_swap_service.destination_images)} destination images loaded."
            )
            self.ready = True
            return True
        except ModelLoadError as e:
            logger.error(f"Failed to load face swap models: {e}")
            self.ready = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}")
            self.ready = False
            return False

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Main prediction method - perform face swap"""
        logger.info("predict() method called")

        if not self.ready:
            return {
                "success": False,
                "error": "Model not ready",
                "detail": "Face swap models are not loaded",
            }

        try:
            # Extract parameters from request
            image_b64 = request.get("image")
            source_face_id = request.get("source_face_id", 1)
            dest_face_id = request.get("destination_face_id", 1)

            if not image_b64:
                return {
                    "success": False,
                    "error": "Missing required parameter: image",
                    "detail": "Request must contain base64 encoded image",
                }

            # Decode base64 image to bytes
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as e:
                return {
                    "success": False,
                    "error": "Invalid image data",
                    "detail": f"Could not decode base64 image: {str(e)}",
                }

            # Process face swap using existing service
            results, faces_detected = self.face_swap_service.process_face_swap_request(
                source_image_data=image_bytes,
                source_face_id=source_face_id,
                dest_face_id=dest_face_id,
            )

            # Format response to match FastAPI schema
            swapped_images = [
                {"image_data": base64_data, "destination_name": filename}
                for base64_data, filename in results
            ]

            response = {
                "success": True,
                "message": f"Successfully swapped face onto {len(swapped_images)} images",
                "swapped_images": swapped_images,
                "faces_detected_in_source": faces_detected,
            }

            logger.info(f"Face swap completed: {len(swapped_images)} images processed")
            return response

        except NoFaceDetectedError as e:
            logger.warning(f"No face detected: {e}")
            return {"success": False, "error": "No face detected", "detail": str(e)}
        except InsufficientFacesError as e:
            logger.warning(f"Insufficient faces: {e}")
            return {"success": False, "error": "Insufficient faces", "detail": str(e)}
        except InvalidImageError as e:
            logger.warning(f"Invalid image: {e}")
            return {"success": False, "error": "Invalid image", "detail": str(e)}
        except FaceSwapError as e:
            logger.error(f"Face swap error: {e}")
            return {"success": False, "error": "Face swap failed", "detail": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "success": False,
                "error": "Internal server error",
                "detail": f"Unexpected error: {str(e)}",
            }

    def preprocess(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Preprocess incoming request"""
        logger.info("preprocess() method called")

        # TODO: In next iteration, handle multipart form data conversion
        # - Extract image from base64 or binary data
        # - Validate image format and size
        # - Extract face IDs from request

        logger.info("Preprocessing completed")
        return request

    def postprocess(
        self, result: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Postprocess prediction results"""
        logger.info("postprocess() method called")

        # TODO: In next iteration, format response for client
        # - Convert base64 images to appropriate format
        # - Add metadata about processing
        # - Format according to API specification

        logger.info("Postprocessing completed")
        return result


if __name__ == "__main__":
    # This is how KServe starts the model server
    model = KServeFaceSwapModel("swaparoony-face-swap")

    # Explicitly load the model
    logger.info("Loading model...")
    if model.load():
        logger.info("Model loaded successfully, starting server...")
        ModelServer().start([model])
    else:
        logger.error("Failed to load model, exiting...")
        exit(1)
