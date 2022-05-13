#!/bin/sh
set -e

case "$ARCHFLAGS" in
    *x86_64)
        ln -sf amd64 vulkan-sdk/current
        ;;
    *arm64)
        ln -sf arm64 vulkan-sdk/current
        ;;
    *)
        echo "Cannot detect arch from $ARCHFLAGS"
        exit 1
esac
