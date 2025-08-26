# Local Testing Environment Setup

This guide provides step-by-step installation instructions for setting up a local Kubernetes testing environment on Fedora 42 and Aurora/Bluefin systems.

## Prerequisites

### Install Git LFS (for model files)
```bash
# Install Git LFS on Fedora 42
sudo dnf install git-lfs

# Verify installation
git lfs version
```

### Ensure Docker is installed and running:
```bash
# Check Docker status
sudo systemctl status docker
docker --version

# Start Docker if not running
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

## 1. Install Kind v0.29.0 (Latest)

Kind (Kubernetes in Docker) provides a lightweight Kubernetes cluster for local development:

```bash
cd Downloads
# Download Kind v0.29.0
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-amd64

# Make executable and install
chmod +x ./kind
sudo mv ./kind ~/bin/kind

# Verify installation
kind version
```

## 2. Install kubectl v1.33.4 (Latest Stable)

kubectl is the command-line tool for interacting with Kubernetes clusters:

```bash
# Download kubectl v1.33.4
curl -LO "https://dl.k8s.io/release/v1.33.4/bin/linux/amd64/kubectl"

# Validate the binary (optional but recommended)
curl -LO "https://dl.k8s.io/v1.33.4/bin/linux/amd64/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check

# Make executable and install
chmod +x kubectl
sudo mv kubectl ~/bin/kubectl

# Verify installation
kubectl version --client
```

## 3. Install Helm v3.18.6 (Latest)

Helm is the package manager for Kubernetes:

### Method 1: Using the official script (recommended)
```bash
# Download and install Helm using official script
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
rm get_helm.sh
```

### Method 2: Direct download
```bash
# Download Helm v3.18.6 directly
curl -LO https://get.helm.sh/helm-v3.18.6-linux-amd64.tar.gz
tar -zxvf helm-v3.18.6-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm

# Clean up
rm -rf linux-amd64/ helm-v3.18.6-linux-amd64.tar.gz
```

```bash
# Verify installation
helm version
```

## 4. Install k9s v0.50.9 (Latest)

k9s provides a terminal-based UI for managing Kubernetes clusters:

```bash
# Download k9s v0.50.9
curl -LO https://github.com/derailed/k9s/releases/download/v0.50.9/k9s_Linux_amd64.tar.gz

# Extract and install
tar -zxvf k9s_Linux_amd64.tar.gz
sudo mv k9s ~/bin/k9s

# Clean up
rm k9s_Linux_amd64.tar.gz LICENSE README.md

# Verify installation
k9s version
```

## 5. Verify Complete Installation

Run all version checks to ensure everything is installed correctly:

```bash
echo "=== Installation Verification ==="
echo "Docker: $(docker --version)"
echo "Kind: $(kind version)"
echo "kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
echo "Helm: $(helm version --short)"
echo "k9s: $(k9s version --short)"
```

## 6. Create Your First Kind Cluster

Create a basic Kind cluster for testing:

```bash
# Create a simple cluster
kind create cluster --name swaparoony-test

# Verify cluster is running
kubectl cluster-info --context kind-swaparoony-test

# List nodes
kubectl get nodes

# Test with k9s (press 'q' to quit)
k9s
```

## 7. Deploy KServe to Your Kind Cluster

Deploy KServe infrastructure to the cluster created in Step 6:

```bash
# Ensure you're using the correct cluster context
kubectl config use-context kind-swaparoony-test

# Install cert-manager (required for KServe)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.15.3/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=available --timeout=300s deployment/cert-manager -n cert-manager
kubectl wait --for=condition=available --timeout=300s deployment/cert-manager-cainjector -n cert-manager
kubectl wait --for=condition=available --timeout=300s deployment/cert-manager-webhook -n cert-manager

# Install KServe (v0.14.0 works best with Kind)
# Note: You may see errors about "Too long: may not be more than 262144 bytes" 
# These are warnings about large CRDs and can be safely ignored
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.0/kserve.yaml

# Wait for KServe controller to be ready
kubectl wait --for=condition=available --timeout=300s deployment/kserve-controller-manager -n kserve

# Verify KServe installation - should see InferenceService CRD
kubectl get crd inferenceservices.serving.kserve.io
kubectl get pods -n kserve
kubectl get crd | grep serving.kserve.io
```

## 8. Build and Deploy the Swaparoony KServe Model

Build the KServe container and deploy the model:

```bash
# Navigate to the repository root
cd /path/to/swaparoony

# Build the KServe container image and load into Kind
docker build -f Dockerfile.kserve -t swaparoony-kserve:latest .
kind load docker-image swaparoony-kserve:latest --name swaparoony-test

# Verify the image is loaded in Kind
docker exec -it swaparoony-test-control-plane crictl images | grep swaparoony

# Deploy the InferenceService
kubectl apply -f deploy/swaparoony-inference-service.yaml

# Monitor the deployment
kubectl get inferenceservices
kubectl get pods

# Check pod logs for initialization
kubectl logs -l serving.kserve.io/inferenceservice=swaparoony-face-swap

# Wait for the service to be ready
kubectl wait --for=condition=ready --timeout=300s inferenceservice/swaparoony-face-swap
```

## 9. Test the KServe Model Deployment

Test the deployed model with the minimal logging implementation:

```bash
# Port forward to access the service locally
kubectl port-forward service/swaparoony-face-swap-predictor-default 8080:80 &

# Test the health endpoint
curl http://localhost:8080/v1/models/swaparoony-face-swap

# Test a basic prediction (should return dummy response)
curl -X POST http://localhost:8080/v1/models/swaparoony-face-swap:predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "image": "test_image_data",
        "source_face_id": 1,
        "destination_face_id": 1
      }
    ]
  }'

# Check the logs to verify method calls
kubectl logs -l serving.kserve.io/inferenceservice=swaparoony-face-swap --tail=20

# Stop port forwarding when done
pkill -f "kubectl port-forward"
```

## 10. Monitor with k9s

Use k9s for easier cluster monitoring:

```bash
# Start k9s
k9s

# Key commands in k9s:
# :pods - View all pods
# :svc - View services
# :inferenceservices - View KServe InferenceServices
# Enter on a pod - View pod details and logs
# l - View logs for selected resource
# d - Describe selected resource
# :q or Ctrl+C - Quit
```

## 11. Troubleshooting Common Issues

### Container Build Failures

**Issue**: `error: command 'g++' failed: No such file or directory`
```bash
# Solution: The Dockerfile.kserve has been updated with required dependencies
# If you see this error, ensure you're using the latest Dockerfile.kserve
docker build -f Dockerfile.kserve -t swaparoony-kserve:latest . --no-cache
```

### InferenceService Deployment Issues

**Issue**: `no matches for kind "InferenceService"`
```bash
# Solution: Install KServe with server-side apply
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.0/kserve.yaml

# Verify InferenceService CRD exists
kubectl get crd inferenceservices.serving.kserve.io
```

**Issue**: `ServerlessModeRejected: It is not possible to use Serverless deployment mode`
```bash
# Solution: The InferenceService YAML includes RawDeployment annotation
# Verify the annotation exists in deploy/swaparoony-inference-service.yaml:
grep "deploymentMode.*RawDeployment" deploy/swaparoony-inference-service.yaml
```

**Issue**: `storage-initializer` fails with "Cannot recognize storage type"
```bash
# Solution: The InferenceService YAML disables storage-initializer
# Verify the annotation exists:
grep "disable-initializer.*true" deploy/swaparoony-inference-service.yaml
```

**Issue**: `NoModelReady: Model with name swaparoony-face-swap is not ready`
```bash
# Solution: The KServe model explicitly calls load() method
# Check pod logs for model loading:
kubectl logs -l serving.kserve.io/inferenceservice=swaparoony-face-swap | grep "Model loaded successfully"
```

### Log Collection

**Issue**: Can't see startup logs or logs are rotating out
```bash
# Restart pod and capture logs from beginning
kubectl delete pod -l serving.kserve.io/inferenceservice=swaparoony-face-swap
kubectl logs -f -l serving.kserve.io/inferenceservice=swaparoony-face-swap --all-containers=true | tee startup-logs.txt

# For continuous log collection during testing
kubectl logs -f -l serving.kserve.io/inferenceservice=swaparoony-face-swap --timestamps=true > test-logs.txt &
# Run your tests, then kill the background job
```

### Testing the Deployment

**Issue**: Want to verify complete request workflow
```bash
# Start port forwarding
kubectl port-forward service/swaparoony-face-swap-predictor-default 8080:80 &

# Test health endpoint
curl http://localhost:8080/v1/models/swaparoony-face-swap

# Test prediction endpoint - should see preprocess→predict→postprocess in logs
curl -X POST http://localhost:8080/v1/models/swaparoony-face-swap:predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "image": "test_image_data",
        "source_face_id": 1,
        "destination_face_id": 1
      }
    ]
  }'

# Check logs for complete workflow
kubectl logs -l serving.kserve.io/inferenceservice=swaparoony-face-swap --tail=20
```

## 12. Cleanup (When Done Testing)

```bash
# Delete the InferenceService
kubectl delete -f deploy/swaparoony-inference-service.yaml

# Delete KServe (optional)
kubectl delete -f https://github.com/kserve/kserve/releases/download/v0.14.0/kserve.yaml

# Delete cert-manager (optional)
kubectl delete -f https://github.com/cert-manager/cert-manager/releases/download/v1.15.3/cert-manager.yaml

# Delete the entire cluster
kind delete cluster --name swaparoony-test

# List remaining clusters
kind get clusters
```

## Platform-Specific Notes

### Fedora 42
- All commands should work without modification
- Ensure SELinux is properly configured for Docker

### Aurora/Bluefin (KDE/Immutable OS)
- All tools install to `/usr/local/bin/` which persists across updates
- Docker should be installed via Distrobox or as a container
- Consider using `toolbox` or `distrobox` for isolated development environments

## Troubleshooting

### Docker Permission Issues
```bash
# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER

# Or run with sudo temporarily
sudo docker run hello-world
```

### Kind Cluster Creation Issues
```bash
# Check Docker is running
systemctl --user status docker

# Clear any existing clusters
kind delete clusters --all

# Try with explicit config
kind create cluster --config=kind-config.yaml
```

### kubectl Connection Issues
```bash
# Check current context
kubectl config current-context

# List available contexts
kubectl config get-contexts

# Switch to Kind context
kubectl config use-context kind-swaparoony-test
```

## Next Steps

Once you've completed the local testing setup above, you can proceed to:
1. Integrate the existing FaceSwapService with the KServe model wrapper
2. Implement interface translation between KServe JSON API and existing FastAPI multipart uploads
3. Handle base64 image conversion and multiple destination images in the KServe model
4. Test the complete face swap functionality locally before production deployment