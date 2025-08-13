import pytest
import numpy as np
import base64
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.swaparoony.services.face_swap_service import FaceSwapService
from src.swaparoony.core.exceptions import (
    NoFaceDetectedError,
    InsufficientFacesError,
    InvalidImageError,
    ModelLoadError,
)


class TestFaceSwapService:
    """Comprehensive tests for FaceSwapService"""

    @pytest.fixture
    def service(self):
        """Create a FaceSwapService instance for testing"""
        return FaceSwapService()

    @pytest.fixture
    def mock_face(self):
        """Create a mock face object"""
        face = Mock()
        face.bbox = [100, 100, 200, 200]  # x, y, w, h
        return face

    @pytest.fixture
    def sample_image(self):
        """Create a sample numpy image array"""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes"""
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00"

    def test_init(self, service):
        """Test service initialization"""
        assert service.app is None
        assert service.swapper is None
        assert service.destination_images == []
        assert service._initialized is False

    @patch("src.swaparoony.services.face_swap_service.FaceAnalysis")
    @patch("src.swaparoony.services.face_swap_service.insightface.model_zoo.get_model")
    @patch("src.swaparoony.services.face_swap_service.cv2.imread")
    @patch("src.swaparoony.services.face_swap_service.Path.exists")
    def test_initialize_models_success(
        self,
        mock_path_exists,
        mock_imread,
        mock_get_model,
        mock_face_analysis,
        service,
        mock_settings,
    ):
        """Test successful model initialization"""
        # Setup mocks
        mock_app = Mock()
        mock_face_analysis.return_value = mock_app
        mock_swapper = Mock()
        mock_get_model.return_value = mock_swapper
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_path_exists.return_value = True  # Mock path exists check

        # Test initialization
        service.initialize_models()

        # Verify models were loaded
        assert service.app == mock_app
        assert service.swapper == mock_swapper
        assert service._initialized is True
        assert len(service.destination_images) == 2

        # Verify method calls
        mock_face_analysis.assert_called_once_with(name="buffalo_l")
        mock_app.prepare.assert_called_once_with(ctx_id=0, det_size=(640, 640))
        mock_get_model.assert_called_once_with(
            "models/test.onnx", download=True, download_zip=True
        )

    @patch("src.swaparoony.services.face_swap_service.FaceAnalysis")
    def test_initialize_models_failure(self, mock_face_analysis, service):
        """Test model initialization failure"""
        mock_face_analysis.side_effect = Exception("Model load failed")

        with pytest.raises(ModelLoadError, match="Failed to initialize models"):
            service.initialize_models()

        assert service._initialized is False

    @patch("src.swaparoony.services.face_swap_service.cv2.imread")
    @patch("src.swaparoony.services.face_swap_service.Path.exists")
    def test_load_destination_images_success(
        self, mock_path_exists, mock_imread, service, mock_settings
    ):
        """Test successful destination image loading"""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_path_exists.return_value = True

        service._load_destination_images()

        assert len(service.destination_images) == 2
        assert all(isinstance(img, np.ndarray) for img, _ in service.destination_images)

    @patch("src.swaparoony.services.face_swap_service.cv2.imread")
    @patch("src.swaparoony.services.face_swap_service.Path.exists")
    def test_load_destination_images_none_found(
        self, mock_path_exists, mock_imread, service, mock_settings
    ):
        """Test failure when no destination images can be loaded"""
        mock_path_exists.return_value = False

        with pytest.raises(
            ModelLoadError, match="No destination images could be loaded"
        ):
            service._load_destination_images()

    def test_ensure_initialized_not_initialized(self, service):
        """Test _ensure_initialized when not initialized"""
        with pytest.raises(ModelLoadError, match="Models not initialized"):
            service._ensure_initialized()

    def test_ensure_initialized_success(self, service):
        """Test _ensure_initialized when initialized"""
        service._initialized = True
        service._ensure_initialized()  # Should not raise

    @patch("src.swaparoony.services.face_swap_service.cv2.imdecode")
    def test_decode_image_success(self, mock_imdecode, service, sample_image_bytes):
        """Test successful image decoding"""
        mock_imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service._decode_image(sample_image_bytes)

        assert isinstance(result, np.ndarray)
        mock_imdecode.assert_called_once()

    @patch("src.swaparoony.services.face_swap_service.cv2.imdecode")
    def test_decode_image_failure(self, mock_imdecode, service, sample_image_bytes):
        """Test image decoding failure"""
        mock_imdecode.return_value = None

        with pytest.raises(InvalidImageError, match="Could not decode image"):
            service._decode_image(sample_image_bytes)

    @patch("src.swaparoony.services.face_swap_service.cv2.imencode")
    def test_encode_image_success(self, mock_imencode, service, sample_image):
        """Test successful image encoding"""
        mock_buffer = np.array([1, 2, 3, 4])
        mock_imencode.return_value = (True, mock_buffer)

        result = service._encode_image(sample_image)

        assert isinstance(result, str)
        expected = base64.b64encode(mock_buffer).decode("utf-8")
        assert result == expected

    def test_get_faces_sorting(self, service, mock_face):
        """Test face detection and sorting"""
        service.app = Mock()

        # Create faces with different x coordinates
        face1 = Mock()
        face1.bbox = [200, 100, 300, 200]
        face2 = Mock()
        face2.bbox = [100, 100, 200, 200]

        service.app.get.return_value = [face1, face2]

        result = service._get_faces(np.zeros((100, 100, 3)))

        # Should be sorted by x coordinate (face2 first)
        assert len(result) == 2
        assert result[0].bbox[0] == 100
        assert result[1].bbox[0] == 200

    def test_validate_face_index_no_faces(self, service):
        """Test validation when no faces detected"""
        with pytest.raises(
            NoFaceDetectedError, match="No faces detected in source image"
        ):
            service._validate_face_index([], 1, "source")

    def test_validate_face_index_insufficient_faces(self, service, mock_face):
        """Test validation when requesting non-existent face"""
        faces = [mock_face]

        with pytest.raises(
            InsufficientFacesError,
            match="source image contains 1 faces, but requested face 3",
        ):
            service._validate_face_index(faces, 3, "source")

    def test_validate_face_index_invalid_index(self, service, mock_face):
        """Test validation with invalid face index (< 1)"""
        faces = [mock_face]

        with pytest.raises(InsufficientFacesError):
            service._validate_face_index(faces, 0, "source")

    def test_validate_face_index_success(self, service, mock_face):
        """Test successful face index validation"""
        faces = [mock_face, mock_face]

        # Should not raise
        service._validate_face_index(faces, 1, "source")
        service._validate_face_index(faces, 2, "source")

    @patch.object(FaceSwapService, "_get_faces")
    def test_swap_face_on_image_success(
        self, mock_get_faces, service, sample_image, mock_face
    ):
        """Test successful face swapping"""
        service._initialized = True
        service.swapper = Mock()

        # Mock face detection
        mock_get_faces.side_effect = [[mock_face], [mock_face]]  # source, dest faces

        # Mock swapper result
        expected_result = np.ones((100, 100, 3), dtype=np.uint8)
        service.swapper.get.return_value = expected_result

        result = service.swap_face_on_image(sample_image, sample_image, 1, 1)

        assert np.array_equal(result, expected_result)
        service.swapper.get.assert_called_once()

    @patch.object(FaceSwapService, "_get_faces")
    def test_swap_face_on_image_no_source_face(
        self, mock_get_faces, service, sample_image
    ):
        """Test face swapping with no source face"""
        service._initialized = True
        mock_get_faces.return_value = []  # No faces detected

        with pytest.raises(NoFaceDetectedError):
            service.swap_face_on_image(sample_image, sample_image, 1, 1)

    @patch.object(FaceSwapService, "_decode_image")
    @patch.object(FaceSwapService, "_get_faces")
    @patch.object(FaceSwapService, "swap_face_on_image")
    @patch.object(FaceSwapService, "_encode_image")
    def test_process_face_swap_request_success(
        self,
        mock_encode,
        mock_swap,
        mock_get_faces,
        mock_decode,
        service,
        sample_image_bytes,
        mock_face,
    ):
        """Test successful face swap request processing"""
        service._initialized = True
        service.destination_images = [(np.zeros((100, 100, 3)), "dest1.jpg")]

        # Setup mocks
        mock_decode.return_value = np.zeros((100, 100, 3))
        mock_get_faces.return_value = [mock_face]  # One face detected
        mock_swap.return_value = np.ones((100, 100, 3))
        mock_encode.return_value = "base64_encoded_image"

        results, faces_count = service.process_face_swap_request(
            sample_image_bytes, 1, 1
        )

        assert len(results) == 1
        assert results[0] == ("base64_encoded_image", "dest1.jpg")
        assert faces_count == 1

    @patch.object(FaceSwapService, "_decode_image")
    @patch.object(FaceSwapService, "_get_faces")
    def test_process_face_swap_request_invalid_source(
        self, mock_get_faces, mock_decode, service, sample_image_bytes
    ):
        """Test face swap request with invalid source face"""
        service._initialized = True

        mock_decode.return_value = np.zeros((100, 100, 3))
        mock_get_faces.return_value = []  # No faces

        with pytest.raises(NoFaceDetectedError):
            service.process_face_swap_request(sample_image_bytes, 1, 1)

    @patch.object(FaceSwapService, "_decode_image")
    @patch.object(FaceSwapService, "_get_faces")
    @patch.object(FaceSwapService, "swap_face_on_image")
    @patch.object(FaceSwapService, "_encode_image")
    def test_process_face_swap_request_partial_failure(
        self,
        mock_encode,
        mock_swap,
        mock_get_faces,
        mock_decode,
        service,
        sample_image_bytes,
        mock_face,
    ):
        """Test face swap request with some destinations failing"""
        service._initialized = True
        service.destination_images = [
            (np.zeros((100, 100, 3)), "dest1.jpg"),
            (np.zeros((100, 100, 3)), "dest2.jpg"),
        ]

        # Setup mocks
        mock_decode.return_value = np.zeros((100, 100, 3))
        mock_get_faces.return_value = [mock_face]

        # First swap succeeds, second fails
        mock_swap.side_effect = [np.ones((100, 100, 3)), Exception("Swap failed")]
        mock_encode.return_value = "base64_encoded_image"

        results, faces_count = service.process_face_swap_request(
            sample_image_bytes, 1, 1
        )

        # Should only return successful swaps
        assert len(results) == 1
        assert results[0] == ("base64_encoded_image", "dest1.jpg")
        assert faces_count == 1

    def test_not_initialized_error_propagation(self, service):
        """Test that methods properly check initialization"""
        methods_requiring_init = [
            lambda: service.swap_face_on_image(None, None),
            lambda: service.process_face_swap_request(b"test"),
        ]

        for method in methods_requiring_init:
            with pytest.raises(ModelLoadError, match="Models not initialized"):
                method()

    @pytest.mark.parametrize(
        "face_count,requested_index,should_fail",
        [
            (0, 1, True),  # No faces, requesting first
            (1, 1, False),  # One face, requesting first - OK
            (1, 2, True),  # One face, requesting second - fail
            (2, 1, False),  # Two faces, requesting first - OK
            (2, 2, False),  # Two faces, requesting second - OK
            (2, 3, True),  # Two faces, requesting third - fail
            (1, 0, True),  # Invalid index (0)
            (1, -1, True),  # Invalid index (negative)
        ],
    )
    def test_face_index_validation_scenarios(
        self, service, face_count, requested_index, should_fail
    ):
        """Test various face index validation scenarios"""
        faces = [Mock() for _ in range(face_count)]

        if should_fail:
            with pytest.raises((NoFaceDetectedError, InsufficientFacesError)):
                service._validate_face_index(faces, requested_index, "test")
        else:
            # Should not raise
            service._validate_face_index(faces, requested_index, "test")

    @patch("src.swaparoony.services.face_swap_service.cv2.imdecode")
    def test_decode_image_exception_handling(self, mock_imdecode, service):
        """Test image decoding with various exceptions"""
        mock_imdecode.side_effect = Exception("Decoding error")

        with pytest.raises(InvalidImageError, match="Invalid image format"):
            service._decode_image(b"invalid_data")

    def test_service_state_persistence(self, service):
        """Test that service maintains state correctly"""
        # Initially not initialized
        assert not service._initialized

        # After setting initialized flag
        service._initialized = True
        assert service._initialized

        # State should persist
        service._ensure_initialized()  # Should not raise

        # Can reset state
        service._initialized = False
        with pytest.raises(ModelLoadError):
            service._ensure_initialized()

    @patch.object(FaceSwapService, "_decode_image")
    def test_decode_image_called_correctly(
        self, mock_decode, service, sample_image_bytes
    ):
        """Test that _decode_image is called with correct parameters"""
        service._initialized = True
        service.destination_images = []

        mock_decode.return_value = np.zeros((100, 100, 3))

        with patch.object(service, "_get_faces", return_value=[Mock()]):
            service.process_face_swap_request(sample_image_bytes, 1, 1)

        mock_decode.assert_called_once_with(sample_image_bytes)
