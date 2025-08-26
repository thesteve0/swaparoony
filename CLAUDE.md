
# Claude Development Context

## Project Overview
Swaparoony is a face-swapping API for trade shows using InsightFace, FastAPI, and PyTorch. The application runs in a devcontainer with CUDA support.

## Current Objective
Deploying the existing face swap API to KServe on OpenShift AI using RawDeployment mode (no scale-to-zero needed).

## Completed Work
- Created minimal KServeFaceSwapModel that inherits from kserve.Model
- Added kserve>=0.15.0 to requirements-filtered.txt  
- Built Dockerfile.kserve for containerization
- Created InferenceService YAML for deployment
- Model currently logs method calls (load, predict, preprocess, postprocess) for testing

## Next Steps
1. Test minimal KServe deployment on local minikube/kind cluster
2. Implement interface translation between KServe JSON API and existing FastAPI multipart uploads
3. Integrate existing FaceSwapService with KServe model wrapper
4. Handle base64 image conversion and multiple destination images

## Existing Codebase
- **FaceSwapService** in `src/swaparoony/services/face_swap_service.py` handles core face swap logic
- **FastAPI routes** in `src/swaparoony/api/routes/face_swap.py` use multipart file uploads
- **Models** expect source_face_id, dest_face_id parameters and return multiple base64 images

## Technical Requirements
- Must maintain compatibility with existing FastAPI interface
- KServe model needs to handle JSON input/output format conversion  
- Heavy InsightFace models require proper initialization in load() method
- Target deployment: OpenShift AI with KServe 0.15+

## Key Files
- `src/swaparoony/services/face_swap_service.py` - Core face swap logic
- `src/swaparoony/api/routes/face_swap.py` - FastAPI multipart interface
- `src/swaparoony/services/kserve_model.py` - KServe model wrapper
- `src/swaparoony/core/config.py` - Configuration
- `deploy/swaparoony-inference-service.yaml` - Kubernetes deployment
- `Dockerfile.kserve` - Container build

## Project Overview

Swaparoony is a FastAPI-based face swapping application designed for trade show demonstrations. It uses InsightFace for facial recognition and face swapping, providing both an API interface and a Gradio demo interface.

## Core Architecture

The application follows a clean architecture pattern with clear separation of concerns:


## Development Commands

**Install Dependencies:**
```bash
# Filter dependencies to avoid conflicts with NVIDIA container
python scripts/resolve-dependencies.py requirements-filtered.txt
uv pip install --break-system-packages --system -r requirements-filtered.txt
```

**Run Development Server:**
```bash
# Run FastAPI with hot reload
python -m uvicorn src.swaparoony.main:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python src/swaparoony/main.py
```

**Run Gradio Demo:**
```bash
python examples/app.py
```

**Run Tests:**
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/swaparoony --cov-report=html

# Run specific test file
pytest tests/test_face_swap_service.py

# Run with verbose output
pytest tests/ -v
```

## API Endpoints

**Main Endpoints:**
- `POST /api/v1/swap` - Face swap operation (accepts image upload, face IDs)
- `GET /api/v1/health` - Service health check
- `GET /` - Root endpoint with service information

**Key Parameters:**
- `source_face_id`: Face position in source image (1-based indexing)
- `destination_face_id`: Face position in destination images (1-based indexing)

## Configuration

Configuration is managed through `src/swaparoony/core/config.py` with these key settings:
- `destination_images`: List of paths to destination images (preloaded at startup)
- `face_analysis_name`: InsightFace model name (default: "buffalo_l")
- `model_path`: Path to face swapping model
- `det_size`: Detection size for face analysis
- `ctx_id`: Context ID for GPU/CPU selection

## Face Processing Logic

**Face Detection:**
- Faces are detected using InsightFace's FaceAnalysis
- Faces are sorted by x-coordinate (left to right)
- Face indexing is 1-based for user-friendly API

**Face Swapping:**
- Source face from uploaded image
- Target faces from preloaded destination images
- Uses InsightFace's inswapper model for face replacement
- Handles multiple destination images in single request

## Testing Strategy

The test suite uses pytest with comprehensive mocking:
- `tests/conftest.py` - Test configuration and automatic settings mocking
- `tests/test_face_swap_service.py` - Comprehensive service testing
- Mocks InsightFace models to avoid dependency on actual model files
- Tests cover error conditions, edge cases, and API contract validation

## Container Environment

This project is designed to run in an NVIDIA PyTorch container:
- Uses `uv` package manager for dependency installation
- Dependency filtering via `scripts/resolve-dependencies.py` prevents conflicts
- GPU acceleration for face processing models
- Development container configuration in `.devcontainer/`

## Image Data Flow

1. **Input**: Image uploaded via FastAPI endpoint
2. **Validation**: Image format and size validation
3. **Face Detection**: Extract and sort faces from source image
4. **Face Swapping**: Apply source face to each preloaded destination image
5. **Encoding**: Convert results to base64 for API response
6. **Response**: JSON with base64 images and metadata

## Error Handling

Custom exceptions for clear error messaging:
- `NoFaceDetectedError`: No faces found in image
- `InsufficientFacesError`: Requested face index exceeds detected faces
- `InvalidImageError`: Image format or decoding issues
- `ModelLoadError`: Model initialization failures
- `FaceSwapError`: Face swapping operation failures

## Performance Considerations

- Destination images preloaded at startup (memory vs. disk I/O trade-off)
- Face models initialized once and reused
- GPU acceleration when available
- Efficient face sorting and indexing
- Base64 encoding for web-compatible image transport