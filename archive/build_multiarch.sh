#!/usr/bin/env bash
# Multi-Architecture Docker Build Script
# Works on: AMD64, ARM64 (Apple Silicon, NVIDIA Grace, Raspberry Pi, etc.)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Multi-Architecture Docker Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect current architecture
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64)  NATIVE_PLATFORM="linux/amd64" ;;
  aarch64|arm64) NATIVE_PLATFORM="linux/arm64" ;;
  *)
    echo -e "${RED}Unknown architecture: ${ARCH}${NC}"
    exit 1
    ;;
esac

echo -e "${GREEN}✓${NC} Detected architecture: ${ARCH} (${NATIVE_PLATFORM})"
echo ""

# Check Docker permissions
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker permission denied!${NC}"
    echo ""
    echo "Fix this by adding your user to the docker group:"
    echo -e "${YELLOW}  sudo usermod -aG docker \$USER${NC}"
    echo -e "${YELLOW}  newgrp docker${NC}"
    echo ""
    echo "Or run this script with sudo (not recommended for security):"
    echo -e "${YELLOW}  sudo $0${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker permissions OK"

# Check buildx
if ! docker buildx version >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker buildx not found!${NC}"
    echo "Install it from: https://github.com/docker/buildx"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker buildx available"
echo ""

# Choose build mode
echo "Choose build mode:"
echo "  1) Native only (${NATIVE_PLATFORM}) - FAST, for local testing"
echo "  2) Multi-arch (amd64 + arm64) - for distribution"
echo "  3) AMD64 only - maximum compatibility (works everywhere via emulation)"
echo ""
read -p "Enter choice [1-3]: " BUILD_MODE

IMAGE_NAME="${IMAGE_NAME:-phire-topo:latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.topo.multiarch}"

case "${BUILD_MODE}" in
  1)
    echo -e "${YELLOW}Building for ${NATIVE_PLATFORM} only...${NC}"
    docker buildx build \
      --platform "${NATIVE_PLATFORM}" \
      -f "${DOCKERFILE}" \
      -t "${IMAGE_NAME}" \
      --load \
      .
    ;;
  2)
    echo -e "${YELLOW}Building for linux/amd64 and linux/arm64...${NC}"
    echo -e "${RED}NOTE: This will PUSH to a registry. Set IMAGE_NAME to your registry.${NC}"
    echo -e "Current IMAGE_NAME: ${IMAGE_NAME}"
    read -p "Continue? [y/N]: " CONFIRM
    if [[ "${CONFIRM}" != "y" ]]; then
      echo "Cancelled."
      exit 0
    fi
    docker buildx build \
      --platform linux/amd64,linux/arm64 \
      -f "${DOCKERFILE}" \
      -t "${IMAGE_NAME}" \
      --push \
      .
    ;;
  3)
    echo -e "${YELLOW}Building for linux/amd64 only...${NC}"
    docker buildx build \
      --platform linux/amd64 \
      -f "${DOCKERFILE}" \
      -t "${IMAGE_NAME}" \
      --load \
      .
    ;;
  *)
    echo -e "${RED}Invalid choice${NC}"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}✓ Build complete!${NC}"
echo ""
echo "Test the image:"
echo -e "${YELLOW}  docker run --rm ${IMAGE_NAME}${NC}"
echo ""
echo "Or run your analysis:"
echo -e "${YELLOW}  docker run --rm -v \"\$PWD\":/work ${IMAGE_NAME} python phase_b_persistence_eval.py${NC}"
