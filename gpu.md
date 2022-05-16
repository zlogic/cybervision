# Vulkan

To build with Vulkan, set the `VULKAN_SDK` environment variable.

## MoltenVK

Note: MoltenVK remains an option, but using Metal is a better option for Apple machines. 

Download [molten-vk](https://formulae.brew.sh/formula/molten-vk) from Homebrew.

To set a custom logging level, use the `MVK_CONFIG_LOG_LEVEL` evironment variable (e.g. `export MVK_CONFIG_LOG_LEVEL=NONE`).

## Metal

The Metal option was only tested on an M1 Macbook Air.

Set the `OBJC_DEBUG_MISSING_POOLS` environment variable to `YES` to confirm that pointer autoreleases are working as expected.

# Compile shaders

Download the [glslang](https://github.com/KhronosGroup/glslang) compiler from [Homebrew](https://formulae.brew.sh/formula/glslang).

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
