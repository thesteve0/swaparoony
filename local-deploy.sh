# Rebuild with the fixed imports
  docker build -f Dockerfile.kserve -t swaparoony-kserve:latest .

  # Load into Kind
  kind load docker-image swaparoony-kserve:latest --name swaparoony-test

  # Redeploy to test the fix
  kubectl delete inferenceservice swaparoony-face-swap
  # kubectl apply -f deploy/swaparoony-inference-service.yaml
  kubectl apply -f deploy/openshift-deploy.yaml
