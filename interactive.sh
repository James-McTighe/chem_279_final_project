#!/bin/sh

docker run --rm -it \
    -v "$(pwd):/work" \
    x79project:latest \
    bash