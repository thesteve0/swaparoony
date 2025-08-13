import sys
import pytest
from pathlib import Path

# Add the src directory to Python path for testing
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


# Mock settings to prevent loading real config during tests
@pytest.fixture(autouse=True)
def mock_settings():
    """Automatically mock settings for all tests"""
    from unittest.mock import patch, MagicMock

    mock_settings = MagicMock()
    mock_settings.face_analysis_name = "buffalo_l"
    mock_settings.ctx_id = 0
    mock_settings.det_size = (640, 640)
    mock_settings.model_path = "models/test.onnx"
    mock_settings.destination_images = ["test1.jpg", "test2.jpg"]
    mock_settings.max_file_size = 2 * 1024 * 1024
    mock_settings.allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    with patch("src.swaparoony.services.face_swap_service.settings", mock_settings):
        with patch("src.swaparoony.core.config.settings", mock_settings):
            yield mock_settings
