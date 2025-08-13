from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # Model paths
    model_path: str = "models/inswapper_128.onnx"
    face_analysis_name: str = "buffalo_l"

    # Detection settings
    ctx_id: int = 0
    det_size: tuple = (640, 640)

    # Destination images for face swapping
    destination_images: List[str] = [
        "data/photos-for-ai/destination/224651.jpg",
        "data/photos-for-ai/destination/Bronzino_-_Portrait_of_a_Young_Man,_1550-1555.jpg",
        "data/photos-for-ai/destination/Hubert_Humphrey_Portrait_Colorized.jpg",
        "data/photos-for-ai/destination/10591115146_c8772afc14_b.jpg",
    ]

    # API settings
    max_file_size: int = 2 * 1024 * 1024  # 2MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp"]

    # Performance
    max_concurrent_requests: int = 6

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
