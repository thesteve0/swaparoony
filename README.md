# Swaparoony 🎭

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![KServe](https://img.shields.io/badge/KServe-0.15+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Swaparoony** is a high-performance face-swapping API service designed for interactive applications and demonstrations. Built on InsightFace and FastAPI, it provides both standalone and cloud-native deployment options with KServe integration for scalable ML serving.

## ✨ Features

- 🚀 **Real-time face swapping** using state-of-the-art InsightFace models
- 🎯 **Multi-face support** with precise face selection (1-based indexing)
- 🌐 **Dual deployment modes**: Standalone FastAPI and KServe cloud-native
- 🖼️ **Multiple output formats** with preloaded destination images
- 🔧 **GPU acceleration** support (CUDA and Intel Arc)
- 📊 **Comprehensive error handling** and validation
- ⚡ **High throughput** with async processing
- 🎨 **Web frontend** via [swaparoony-frontend](https://github.com/thesteve0/swaparoony-frontend)

## 🏗️ Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Frontend App      │    │   Swaparoony API    │    │  InsightFace Models │
│                     │    │                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │   React UI      │◄├────┤ │   FastAPI       │ │    │ │   buffalo_l     │ │
│ │                 │ │    │ │   Routes        │ │    │ │  (face analysis)│ │
│ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │
│                     │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ ┌─────────────────┐ │    │ │ FaceSwapService │◄├────┤ │ inswapper_128   │ │
│ │   File Upload   │ │    │ │                 │ │    │ │  (face swap)    │ │
│ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │
│                     │    │ ┌─────────────────┐ │    │                     │
└─────────────────────┘    │ │ KServe Wrapper  │ │    └─────────────────────┘
                           │ │  (Optional)     │ │    
                           │ └─────────────────┘ │    ┌─────────────────────┐
                           └─────────────────────┘    │ Destination Images  │
                                                      │                     │
                              Deployment Options:     │ ┌─────────────────┐ │
                           ┌─────────────────────┐    │ │  portrait1.jpg  │ │
                           │   Standalone        │    │ │  portrait2.jpg  │ │
                           │   FastAPI Server    │    │ │  portrait3.jpg  │ │
                           └─────────────────────┘    │ │  portrait4.jpg  │ │
                           ┌─────────────────────┐    │ └─────────────────┘ │
                           │  KServe on K8s      │    └─────────────────────┘
                           │  (Production)       │    
                           └─────────────────────┘    
```

### Data Flow

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   User   │───▶│  Upload      │───▶│   Face      │───▶│   Swap to    │
│  Photo   │    │  + Face IDs  │    │ Detection   │    │ Destination  │
└──────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                           │                    │
                                           ▼                    ▼
                                    ┌─────────────┐    ┌──────────────┐
                                    │  Sort by    │    │   Return     │
                                    │  X-Position │    │  Base64      │
                                    │ (L→R order) │    │  Images      │
                                    └─────────────┘    └──────────────┘
```

## 📦 Project Structure

```
├── src/swaparoony/
│   ├── api/                        # FastAPI routes and dependencies  
│   ├── services/                   # Core business logic
│   │   ├── face_swap_service.py    # Face processing engine
│   │   └── kserve_model.py         # KServe model wrapper
│   ├── models/                     # Pydantic schemas
│   ├── core/                       # Configuration and exceptions
│   └── utils/                      # Utility functions
├── data/photos-for-ai/
│   └── destination/                # Target images for face placement
├── models/                         # ML model files (Git LFS)
│   └── inswapper_128.onnx         # Face swapping model (manual download)
├── deploy/                         # Kubernetes deployment manifests
├── tests/                          # Test suite
├── .devcontainer/                  # Development environment
├── scripts/                        # Utility scripts
└── headshot.webp                   # Test image (project author)
```

## 🚀 Quick Start

### Prerequisites

- **Git LFS**: Required for model files
  ```bash
  git lfs install
  ```
- **NVIDIA GPU** (recommended) or Intel Arc GPU for acceleration
- **Docker** with GPU support (for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/swaparoony.git
   cd swaparoony
   ```

2. **Pull LFS files:**
   ```bash
   git lfs pull
   ```

3. **Download the face swapping model:**
   
   ⚠️ **Manual Download Required**: The `inswapper_128.onnx` model must be downloaded manually from one of the many available sources on the internet (search for "inswapper_128.onnx download"). Place it in the `models/` directory.

4. **Install dependencies:**
   ```bash
   python scripts/resolve-dependencies.py requirements-filtered.txt
   pip install -r requirements-filtered.txt
   ```

## 📦 Deployment Options

### Option 1: Standalone FastAPI Server

**Development Mode:**
```bash
# Start with hot reload
python -m uvicorn src.swaparoony.main:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python src/swaparoony/main.py
```

**API Endpoints:**
- `POST /api/v1/swap` - Face swap operation
- `GET /api/v1/health` - Health check
- `GET /` - Service information

### Option 2: KServe Cloud-Native Deployment

**Build KServe Container:**
```bash
docker build -f Dockerfile.kserve -t swaparoony-kserve:latest .
```

**Deploy to Kubernetes:**
```bash
kubectl apply -f deploy/swaparoony-inference-service.yaml
```

**Test KServe Endpoint:**
```bash
# Port forward to access locally
kubectl port-forward service/swaparoony-face-swap-predictor 8080:80

# Test with provided script
python test_kserve_visual.py
```

## 🎮 Frontend Interface

The official web interface is available at [swaparoony-frontend](https://github.com/thesteve0/swaparoony-frontend):

- **React-based UI** for image uploads and face selection
- **Real-time preview** of face swap results
- **Responsive design** for desktop and mobile
- **Gallery view** of generated images

## 🧪 Testing

**Test with provided image:**
```bash
# The project includes headshot.webp (project author's photo) for testing
python test_kserve_visual.py
```

**Run the test suite:**
```bash
# All tests
pytest tests/

# With coverage report
pytest tests/ --cov=src/swaparoony --cov-report=html
```

## 📋 Key File Descriptions

### Core Configuration Files

**`CLAUDE.md`** - Development context and instructions for Claude AI assistant. Contains project overview, completed work, next steps, and development commands. Essential for AI-assisted development.

**`scripts/resolve-dependencies.py`** - Filters Python dependencies to avoid conflicts with the NVIDIA PyTorch container's pre-installed packages. Prevents version conflicts when installing requirements.

### Development Environment

**`.devcontainer/devcontainer.json`** - VS Code devcontainer configuration for NVIDIA CUDA development:
- Uses `nvcr.io/nvidia/pytorch:25.04-py3` base image
- Configures GPU access (`NVIDIA_VISIBLE_DEVICES=all`)
- Sets up Python environment and VS Code extensions
- Mounts model cache volumes for performance

**`.devcontainer/setup-environment.sh`** - Post-creation setup script:
- Installs development tools (uv, black, flake8)
- Configures Git identity and Git LFS
- Extracts NVIDIA-provided packages for dependency resolution
- Sets up workspace permissions

### Deployment Files

**`deploy/swaparoony-inference-service.yaml`** - KServe InferenceService manifest:
- Configures RawDeployment mode (no auto-scaling)
- Sets resource limits (500m-1000m CPU, 5-10Gi memory)
- Defines health check endpoints
- Targets OpenShift AI with KServe 0.15+

**`Dockerfile.kserve`** - Container build file for KServe deployment:
- Multi-stage build for optimized image size
- Includes InsightFace models and dependencies
- Configures KServe model server entrypoint
- GPU-optimized for production inference

### Testing and Utilities

**`test_kserve_visual.py`** - Visual testing script for KServe deployment:
- Loads test image and converts to base64
- Sends requests to KServe prediction endpoint
- Saves response images as WebP files for visual verification
- Includes error handling and timeout management (3 minutes)

### Setup Scripts

**`INSTALL_LOCAL_TEST.md`** - Local installation and testing instructions
**`local-deploy.sh`** - Local deployment automation script  
**`setup-project.sh`** - Project initialization and setup script

## 🔧 Configuration

### Destination Images

Place your target images in `data/photos-for-ai/destination/`. These are the images where faces will be replaced:

```bash
data/photos-for-ai/destination/
├── portrait1.jpg
├── portrait2.jpg  
├── portrait3.jpg
└── portrait4.jpg
```

The service preloads these images at startup for optimal performance.

### GPU Configuration

**CUDA (Default):**
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
```

**Intel Arc GPU Adaptation:**

For Intel Arc GPU support, modify the devcontainer:

1. **Update `.devcontainer/devcontainer.json`:**
   ```json
   {
     "image": "intel/intel-extension-for-pytorch:2.1.10-xpu",
     "runArgs": ["--device=/dev/dri", "--ipc=host"],
     "containerEnv": {
       "SYCL_CACHE_PERSISTENT": "1",
       "USE_XPU": "1"
     }
   }
   ```

2. **Modify `.devcontainer/setup-environment.sh`:**
   ```bash
   # Add Intel XPU support
   pip install intel-extension-for-pytorch mkl
   apt-get install intel-opencl-icd
   ```

3. **Update model configuration:**
   ```python
   # In src/swaparoony/core/config.py
   ctx_id: int = -1  # Use XPU instead of CUDA
   ```

## 📋 API Reference

### FastAPI Interface

**Face Swap Request:**
```python
POST /api/v1/swap
Content-Type: multipart/form-data

{
  "image": <uploaded_file>,
  "source_face_id": 1,        # Face position in source (1-based)
  "destination_face_id": 1    # Face position in destinations (1-based)
}
```

### KServe Interface

**Prediction Request:**
```python
POST /v1/models/swaparoony-face-swap:predict
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "source_face_id": 1,
  "destination_face_id": 1
}
```

**Response Format (both interfaces):**
```json
{
  "success": true,
  "message": "Successfully swapped face onto 4 images",
  "swapped_images": [
    {
      "image_data": "base64_encoded_image",
      "destination_name": "portrait1.jpg"
    }
  ],
  "faces_detected_in_source": 1
}
```

## 🛠️ Model Requirements

### Required Models

- **Face Analysis**: InsightFace `buffalo_l` model (auto-downloaded on first run)
- **Face Swapper**: `models/inswapper_128.onnx` (~256MB) - **Manual download required**

⚠️ **Important**: The `inswapper_128.onnx` model is not included in this repository due to licensing considerations. You must download it manually from one of the many available sources online and place it in the `models/` directory.

### Face Processing Logic

1. **Face Detection**: Uses InsightFace's FaceAnalysis with buffalo_l model
2. **Face Sorting**: Detected faces are sorted left-to-right by x-coordinate
3. **Face Indexing**: 1-based indexing for user-friendly API (1st face, 2nd face, etc.)
4. **Face Swapping**: Source face from uploaded image replaces target face in destination images

## 🔍 Troubleshooting

**Common Issues:**

1. **Missing inswapper model:**
   ```
   Error: Could not load model from models/inswapper_128.onnx
   ```
   Download the model manually and place it in the `models/` directory.

2. **Git LFS models missing:**
   ```bash
   git lfs pull
   ```

3. **CUDA out of memory:**
   - Reduce image size or batch size
   - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

4. **Dependency conflicts:**
   ```bash
   python scripts/resolve-dependencies.py requirements-filtered.txt
   ```

5. **KServe timeout errors:**
   - Increase timeout in client requests (default: 180s)
   - Check pod logs: `kubectl logs -f <pod-name>`

## 🧩 Performance Optimization

**GPU Memory:**
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.6"
```

**Concurrency:**
```python
uvicorn src.swaparoony.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

**Image Processing:**
- Preload destination images at startup
- Use appropriate detection sizes (640x640 default)
- Enable async processing for multiple requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face analysis and swapping models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [KServe](https://kserve.github.io/) for cloud-native model serving
- [NVIDIA PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) container for GPU support

## 📞 Support

For issues and questions:
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-org/swaparoony/issues)
- 💬 **Frontend**: [swaparoony-frontend](https://github.com/thesteve0/swaparoony-frontend)
- 📖 **Documentation**: See `CLAUDE.md` for development context

---

*Built with ❤️ for interactive face-swapping experiences*