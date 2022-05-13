#!/bin/sh

mkdir vulkan-loader vulkan-sdk
cd vulkan-loader
curl -L -o - https://github.com/KhronosGroup/Vulkan-Loader/archive/refs/tags/v1.3.213.tar.gz | tar -xz --strip-components=1

cmake -A x64 -S. -Bbuild -DUPDATE_DEPS=On -DCMAKE_INSTALL_PREFIX=../vulkan-sdk
cmake --build build --config Release --target install

cp -rf external/Vulkan-Headers/build/install/include ../vulkan-sdk/
