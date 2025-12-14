#!/bin/sh

docker build -t x79project:latest .

# This fall back was added for those without Docker installed
if [ $? -ne 0 ]; then
  podman build -t x79project:latest .
fi

