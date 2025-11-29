#!/bin/sh

docker run --rm -it \
    -v "$(pwd):/work" \
    x79project:latest \
    bash

if [ $? -ne 0 ]; then
    podman run --rm -it \
        -v "$(pwd):/work" \
        x79project:latest \
        bash
fi