#!/bin/sh

docker build -t x79project:latest .

if [ $? -ne 0 ]; then
    podman build -t x79project:latest .
fi