#!/bin/bash
VERSION=$(cat VERSION)

echo " Building Docker images with version: $VERSION"
docker build -f api.Dockerfile -t fastapi-backend:$VERSION .
docker build -f ui.Dockerfile -t gradio-ui:$VERSION .
