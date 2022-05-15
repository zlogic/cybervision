#!/bin/sh

mkdir -p molten-vk vulkan-sdk/lib
cd molten-vk
curl -L -o - https://github.com/KhronosGroup/MoltenVK/archive/refs/tags/v1.1.9.tar.gz | tar -xvz --strip-components=1
./fetchDependencies --macos

xcodebuild ARCHS="arm64 x86_64" ONLY_ACTIVE_ARCH=NO MVK_SKIP_DYLIB=NO \
    build -quiet -project MoltenVKPackaging.xcodeproj \
    -scheme "MoltenVK Package (macOS only)" -configuration "Release"

cp -rf External/Vulkan-Headers/include ../vulkan-sdk/
cp -rf Package/Release/MoltenVK/dylib/macOS/libMoltenVK.dylib ../vulkan-sdk/lib/
