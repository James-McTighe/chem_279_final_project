#!/bin/sh

docker run --rm -it \
  -v "$(pwd):/work" \
  x79project:latest \
  bash

# This fall back was added for those without Docker installed
if [ $? -ne 0 ]; then
  podman run --rm -it \
    -v "$(pwd):/work" \
    x79project:latest \
    bash
fi

