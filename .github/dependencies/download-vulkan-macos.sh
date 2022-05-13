#!/bin/sh
set -e

mkdir -p vulkan-sdk/amd64 vulkan-sdk/arm64

brew fetch --bottle-tag=big_sur molten-vk |
    grep -E "(Downloaded to:|Already downloaded:)" |\
    grep -v pkg-config |\
    awk '{ print $3 }' |\
    xargs -I {} mv {} vulkan-sdk/amd64/
brew fetch --bottle-tag=arm64_big_sur molten-vk |
    grep -E "(Downloaded to:|Already downloaded:)" |\
    grep -v pkg-config |\
    awk '{ print $3 }' |\
    xargs -I {} mv {} vulkan-sdk/arm64/
