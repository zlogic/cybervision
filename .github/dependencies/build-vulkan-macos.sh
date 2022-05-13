#!/bin/sh

mkdir -p molten-vk vulkan-sdk/arm64/lib vulkan-sdk/amd64/lib
cd molten-vk
curl -L -o - https://github.com/KhronosGroup/MoltenVK/archive/refs/tags/v1.1.9.tar.gz | tar -xvz --strip-components=1
./fetchDependencies --macos

xcodebuild ARCHS="arm64" \
    build -quiet -project MoltenVKPackaging.xcodeproj \
    -scheme "MoltenVK Package (macOS only)" -configuration "Release"

cp -rf External/Vulkan-Headers/include ../vulkan-sdk/arm64/
cp -rf Package/Release/MoltenVK/dylib/macOS/libMoltenVK.dylib ../vulkan-sdk/arm64/lib/

xcodebuild ARCHS="x86_64" \
    build -quiet -project MoltenVKPackaging.xcodeproj \
    -scheme "MoltenVK Package (macOS only)" -configuration "Release"

cp -rf External/Vulkan-Headers/include ../vulkan-sdk/amd64/
cp -rf Package/Release/MoltenVK/dylib/macOS/libMoltenVK.dylib ../vulkan-sdk/amd64/lib/
