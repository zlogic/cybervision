# Vulkan

## Ubuntu Linux

Install the `libvulkan-dev` package to get the Vulkan SDK.

## Windows

To build with Vulkan, set the `VULKAN_SDK` environment variable.

Install the [Vulkan SDK](https://vulkan.lunarg.com) and set the `VULKAN_SDK` environment variable (should happen as a part of the installation process).

Alternatively, build and install [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader) and set the `VULKAN_SDK` environment variable to point its installation target.

## MoltenVK

Note: MoltenVK remains an option, but using Metal is a better option for Apple machines. 

Download [molten-vk](https://formulae.brew.sh/formula/molten-vk) from Homebrew.

To set a custom logging level, use the `MVK_CONFIG_LOG_LEVEL` evironment variable (e.g. `export MVK_CONFIG_LOG_LEVEL=NONE`).

## Metal

The Metal option was only tested on an M1 Macbook Air and M1 Pro Macbook Pro.

Set the `OBJC_DEBUG_MISSING_POOLS` environment variable to `YES` to confirm that pointer autoreleases are working as expected.
⚠️ Sending `commit` to `commandBuffer` will generate *autoreleased with no pool in place* errors - this appears to be an issue with calling Metal from C code; MoltenVK generates the same errors.

# Compile shaders

In macOS, download the [glslang](https://github.com/KhronosGroup/glslang) compiler from [Homebrew](https://formulae.brew.sh/formula/glslang).

In Ubuntu, install the `glslang-tools` paskage.

Compile the shader and convert it into a C source that can be embedded into the binary.

```shell
glslangValidator -V shaders/correlation.glsl -o shaders/correlation.spv
xxd -i shaders/correlation.spv > machine/shaders_spv.h
rm shaders/correlation.spv
```

Or with the MoltenVKShaderConverter:

```shell
MoltenVKShaderConverter -gi shaders/correlation.glsl -oh -t c -so machine/shaders_spv.h
xxd -i shaders/correlation.metal > machine/shaders_metal.h
```
